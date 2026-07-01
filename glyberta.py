"""
GlyBERTa — a RoBERTa masked language model for glycan IUPAC sequences.

This script:
  1. Builds a glycan-aware tokenizer directly from a corpus of IUPAC sequences
     (one sequence per line).
  2. Splits the corpus into train (90%) and test (10%) sets.
  3. Trains a RoBERTa masked language model (MLM) from scratch.
  4. Evaluates the MLM on the withheld test set (loss, perplexity, masked-token
     accuracy).
  5. Provides a `compare` mode that embeds two IUPAC sequences with the trained
     model and reports their cosine (semantic) similarity.

Tokenization
------------
Glycan IUPAC condensed strings look like:

    Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc

The meaningful units ("glyco-letters") are:
  * monosaccharides            -> Gal, GlcNAc, Man, Neu5Ac, Fuc, ...
  * linkages in parentheses    -> (b1-4), (a2-3), ...
  * branch delimiters          -> [  ]

We isolate these units with a single regular expression that the HuggingFace
`tokenizers` pre-tokenizer applies to the raw string, then learn a WordLevel
vocabulary over them. The result is an interpretable, structure-respecting
tokenizer that needs no hand-written monosaccharide list.

Usage
-----
    # Train + evaluate
    python glyberta.py train --data sequences.txt --output_dir ./glyberta-model

    # Compare two sequences with a trained model
    python glyberta.py compare --output_dir ./glyberta-model \
        --seq1 "Gal(b1-4)GlcNAc" --seq2 "Gal(b1-3)GlcNAc"

    # Re-run evaluation only on the saved test split
    python glyberta.py evaluate --output_dir ./glyberta-model
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch

from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import RobertaProcessing

from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Glyco-letter pre-tokenization
# ---------------------------------------------------------------------------
# Match (and isolate as their own tokens):
#   \([^()]*\)  a linkage group such as (b1-4) or (a2-3) or (?1-?)
#   \[ or \]    a branch delimiter
# Everything *between* matches (the monosaccharide names) is also emitted as a
# token by the "isolated" split behavior. Empty pieces are dropped.
GLYCOLETTER_PATTERN = r"\([^()]*\)|\[|\]"

SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def read_sequences(path):
    """Read non-empty, stripped lines from a file."""
    seqs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                seqs.append(s)
    if not seqs:
        raise ValueError(f"No sequences found in {path}")
    return seqs


def build_tokenizer(train_sequences, max_len):
    """Train a glycan-aware WordLevel tokenizer and wrap it for transformers."""
    backend = Tokenizer(WordLevel(unk_token="<unk>"))
    # The pre-tokenizer turns a raw IUPAC string into glyco-letters.
    backend.pre_tokenizer = Split(
        pattern=Regex(GLYCOLETTER_PATTERN),
        behavior="isolated",
    )

    trainer = WordLevelTrainer(special_tokens=SPECIAL_TOKENS)
    backend.train_from_iterator(train_sequences, trainer=trainer)

    # Add <s> ... </s> around each sequence, RoBERTa-style.
    backend.post_processor = RobertaProcessing(
        sep=("</s>", backend.token_to_id("</s>")),
        cls=("<s>", backend.token_to_id("<s>")),
    )
    backend.enable_truncation(max_length=max_len)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=max_len,
    )
    return tokenizer


class SequenceDataset(torch.utils.data.Dataset):
    """Tokenizes sequences up front; the collator handles padding + masking."""

    def __init__(self, sequences, tokenizer, max_len):
        self.examples = tokenizer(
            sequences,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
        )["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.examples[idx], dtype=torch.long)}


def preprocess_logits_for_metrics(logits, labels):
    """Keep only the argmax so we don't hold full-vocab logits in memory."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """Masked-token top-1 accuracy over positions where a label is present."""
    preds, labels = eval_pred
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    mask = labels != -100
    if mask.sum() == 0:
        return {"masked_accuracy": 0.0}
    correct = (preds[mask] == labels[mask]).sum()
    return {"masked_accuracy": float(correct) / float(mask.sum())}


def split_data(sequences, val_frac, test_frac, seed):
    rng = random.Random(seed)
    idx = list(range(len(sequences)))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(sequences) * test_frac)))
    n_val = max(1, int(round((len(sequences) * val_frac))))
    test_idx = set(idx[:n_test])
    val_idx = set(idx[n_test:(n_test+n_val)])
    train_idx = set(idx[(n_test+n_val):])
    train = [sequences[i] for i in train_idx]
    test = [sequences[i] for i in test_idx]
    val = [sequences[i] for i in val_idx]
    return train, val, test

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    sequences = read_sequences(args.data)
    print(f"Loaded {len(sequences)} sequences from {args.data}")

    train_seqs, val_seqs, test_seqs = split_data(sequences, args.test_frac, args.test_frac, args.seed)
    print(f"Train: {len(train_seqs)}   Validation: {len(val_seqs)}   Test: {len(test_seqs)}")

    # Persist the exact splits for reproducible evaluation.
    with open(os.path.join(args.output_dir, "train.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(train_seqs))
    with open(os.path.join(args.output_dir, "validation.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(val_seqs))
    with open(os.path.join(args.output_dir, "test.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(test_seqs))

    # Tokenizer is learned on TRAIN ONLY to avoid leaking test vocabulary.
    tokenizer = build_tokenizer(train_seqs, args.max_len)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Report how many test glyco-letters are unseen (-> <unk>), as a data check.
    unk_id = tokenizer.convert_tokens_to_ids("<unk>")
    test_ids = tokenizer(test_seqs, add_special_tokens=False)["input_ids"]
    n_unk = sum(tok == unk_id for ids in test_ids for tok in ids)
    n_tok = sum(len(ids) for ids in test_ids)
    if n_tok:
        print(f"Test out-of-vocabulary rate: {n_unk}/{n_tok} = {n_unk / n_tok:.3%}")

    val_ids = tokenizer(val_seqs, add_special_tokens=False)["input_ids"]
    n_unk = sum(tok == unk_id for ids in val_ids for tok in ids)
    n_tok = sum(len(ids) for ids in val_ids)
    if n_tok:
        print(f"Validation out-of-vocabulary rate: {n_unk}/{n_tok} = {n_unk / n_tok:.3%}")

    train_ds = SequenceDataset(train_seqs, tokenizer, args.max_len)
    val_ds = SequenceDataset(val_seqs, tokenizer, args.max_len)
    test_ds = SequenceDataset(test_seqs, tokenizer, args.max_len)

    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        # RoBERTa reserves positions for the padding offset (pad_idx + 1 ...).
        max_position_embeddings=args.max_len + 2,
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = RobertaForMaskedLM(config)
    print(f"Model parameters: {model.num_parameters():,}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    warmup_ratio = 0.05
    total_training_samples = len(train_seqs)
    total_steps = (total_training_samples // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * warmup_ratio)
    training_args = TrainingArguments(
        eval_strategy="epoch",
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        report_to=[],
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    # Save the final model + tokenizer.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = evaluate_model(trainer)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("\nSaved model, tokenizer, and test_metrics.json to", args.output_dir)


def evaluate_model(trainer):
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")
    if loss is not None:
        metrics["perplexity"] = math.exp(loss) if loss < 30 else float("inf")
    print("\n=== Test-set evaluation ===")
    print(f"  loss            : {metrics.get('eval_loss'):.4f}")
    print(f"  perplexity      : {metrics.get('perplexity'):.4f}")
    if "eval_masked_accuracy" in metrics:
        print(f"  masked accuracy : {metrics['eval_masked_accuracy']:.4%}")
    return metrics


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args):
    set_seed(args.seed)
    test_path = os.path.join(args.output_dir, "test.txt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"{test_path} not found — run `train` first, or pass --data."
        )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.output_dir)
    model = RobertaForMaskedLM.from_pretrained(args.output_dir)
    test_seqs = read_sequences(args.data) if args.data else read_sequences(test_path)
    max_len = tokenizer.model_max_length
    test_ds = SequenceDataset(test_seqs, tokenizer, max_len)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    evaluate_model(trainer)


# ---------------------------------------------------------------------------
# compare (semantic similarity)
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed(sequences, tokenizer, model, device):
    """Mean-pool the final hidden states over non-special, non-pad tokens."""
    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)
    out = model.roberta(**enc)  # encoder without the MLM head
    hidden = out.last_hidden_state  # (batch, seq, hidden)

    mask = enc["attention_mask"].clone()
    # Drop special tokens (<s>, </s>, <pad>) from the pooling.
    special_ids = set(tokenizer.all_special_ids)
    for sid in special_ids:
        mask = mask & (enc["input_ids"] != sid).long()

    mask = mask.unsqueeze(-1).type_as(hidden)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cmd_compare(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.output_dir)
    model = RobertaForMaskedLM.from_pretrained(args.output_dir).to(device).eval()

    embs = embed([args.seq1, args.seq2], tokenizer, model, device)
    sim = torch.nn.functional.cosine_similarity(embs[0:1], embs[1:2]).item()

    print(f"seq1: {args.seq1}")
    print(f"seq2: {args.seq2}")
    print(f"\ncosine similarity: {sim:.4f}")

    # Show how each sequence was tokenized — useful for sanity checking.
    for name, seq in (("seq1", args.seq1), ("seq2", args.seq2)):
        toks = tokenizer.tokenize(seq)
        print(f"{name} tokens: {toks}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="GlyBERTa: RoBERTa MLM for glycan IUPAC sequences.")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output_dir", default="./glyberta-model")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--mlm_probability", type=float, default=0.15)
    common.add_argument("--batch_size", type=int, default=32)
    common.add_argument("--max_len", type=int, default=128)

    pt = sub.add_parser("train", parents=[common], help="Train + evaluate the MLM.")
    pt.add_argument("--data", required=True, help="Text file, one IUPAC sequence per line.")
    pt.add_argument("--test_frac", type=float, default=0.10)
    pt.add_argument("--epochs", type=float, default=20)
    pt.add_argument("--learning_rate", type=float, default=5e-4)
    pt.add_argument("--hidden_size", type=int, default=256)
    pt.add_argument("--num_layers", type=int, default=4)
    pt.add_argument("--num_heads", type=int, default=4)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("evaluate", parents=[common], help="Evaluate on the saved test split.")
    pe.add_argument("--data", default=None, help="Optional alternate test file.")
    pe.set_defaults(func=cmd_evaluate)

    pc = sub.add_parser("compare", parents=[common], help="Cosine similarity of two sequences.")
    pc.add_argument("--seq1", required=True)
    pc.add_argument("--seq2", required=True)
    pc.set_defaults(func=cmd_compare)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
