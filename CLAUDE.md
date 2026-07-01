# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working in this repo

Always describe and show the proposed changes (e.g. the before/after code) before asking whether to apply them. Do not edit files until the user has approved.

## What this is

GlyBERTa trains a RoBERTa masked language model (MLM) from scratch over glycan IUPAC-condensed sequences, then uses the learned encoder to measure semantic similarity between glycans. The entire pipeline — tokenizer training, data splitting, model training, evaluation, and comparison — lives in the single script [glyberta.py](glyberta.py). [demonstration.ipynb](demonstration.ipynb) is a Colab-oriented walkthrough of the same flow.

## Commands

```bash
pip install -r requirements.txt          # torch, transformers, tokenizers, numpy

python make_sample_data.py               # writes sample_sequences.txt (400 synthetic glycans)

# Train + evaluate (writes model, tokenizer, splits, test_metrics.json into --output_dir)
python glyberta.py train --data sample_sequences.txt --output_dir ./glyberta-model

# Re-run evaluation only, on the saved test split (or --data an alternate file)
python glyberta.py evaluate --output_dir ./glyberta-model

# Cosine similarity between two sequences using the trained encoder
python glyberta.py compare --output_dir ./glyberta-model \
    --seq1 "Gal(b1-4)GlcNAc" --seq2 "Gal(b1-3)GlcNAc"
```

There is no test suite, linter, or build step. Validate changes by running `train` on `sample_sequences.txt`.

### Programmatic / notebook entry point

`glyberta.run(command)` runs any of the above subcommands from a single string (or a pre-split token list) instead of the shell, e.g. `glyberta.run(f"train --data {path} --epochs 50")`. This is what [demonstration.ipynb](demonstration.ipynb) uses. `main()` (the CLI entry point) is just `run(sys.argv[1:])`, so the CLI and notebook paths are identical. `run` splits with `shlex` (quoted args work), resolves the seed, dispatches to the `cmd_*` handler, and returns the parsed args namespace.

### Seeds and reproducibility

`--seed` defaults to a random value. When omitted, `run()` picks one and prints it as `Random seed: <n>`; when `--seed` is passed explicitly, nothing is printed. Pass `--seed` explicitly to reproduce a prior run. The seed flows into `set_seed`, the train/val/test split, and HuggingFace `TrainingArguments`, and is shared by all subcommands.

## Architecture

The design goal is a **structure-respecting, interpretable tokenizer that needs no hand-written monosaccharide list.**

- **Glyco-letter tokenization** (`build_tokenizer`): a single regex (`GLYCOLETTER_PATTERN`) drives a `tokenizers` `Split` pre-tokenizer with `behavior="isolated"`. It isolates linkage groups `(b1-4)` and branch delimiters `[` `]` as their own tokens; the monosaccharide names between matches (`Gal`, `GlcNAc`, …) fall out as tokens automatically. A `WordLevel` vocab is then learned over these units — no BPE, no subword merging. The backend `Tokenizer` is wrapped in a `PreTrainedTokenizerFast` with RoBERTa-style `<s>…</s>` post-processing.

- **Vocabulary is learned on the training split only** (`cmd_train`), so test/val vocabulary cannot leak into the tokenizer. The script explicitly reports the out-of-vocabulary (`<unk>`) rate on val and test as a data-quality check.

- **Data splitting** (`split_data`): shuffles by seed into train/val/test. Note `cmd_train` calls `split_data(sequences, args.test_frac, args.test_frac, args.seed)` — the same `--test_frac` value is deliberately reused for both the validation and test fractions. The three splits are persisted to `train.txt` / `validation.txt` / `test.txt` in `--output_dir` so `evaluate` can reload the exact test set later.

- **Model & training** (`cmd_train`): a `RobertaForMaskedLM` built fresh from a `RobertaConfig` sized by CLI args (`--hidden_size`, `--num_layers`, `--num_heads`; `intermediate_size` is `hidden_size * 4`; `max_position_embeddings` is `--max_len + 2` for RoBERTa's padding offset). `DataCollatorForLanguageModeling` performs padding and dynamic MLM masking at batch time (`SequenceDataset` only stores token ids). Best-model selection uses **minimum validation loss** (`metric_for_best_model="eval_loss"`, `greater_is_better=False`, `load_best_model_at_end=True`), so the saved checkpoint is the lowest-val-loss epoch, not the last. After training, `cmd_train` recovers the retained epoch by matching `trainer.state.best_metric` against `trainer.state.log_history` and prints it.

- **Metrics**: `preprocess_logits_for_metrics` reduces logits to argmax before they leave the GPU (avoids holding full-vocab logits in memory); `compute_metrics` reports masked-token top-1 accuracy over non-`-100` label positions. `evaluate_model` adds perplexity (`exp(loss)`). Note MLM masking is random per evaluation pass, so a re-run of `trainer.evaluate()` (e.g. the final "Test-set evaluation" block, which actually runs on the validation set) will not exactly equal any per-epoch row even for the same weights.

- **Comparison** (`cmd_compare` / `embed`): embeds sequences with `model.roberta` (encoder only, no MLM head) and mean-pools the final hidden states over real tokens, explicitly masking out special and pad tokens, then reports cosine similarity.

## Versioning

[glyberta.py](glyberta.py) prints a version string at import (`print("GlyBerta v1.0.7")`, near the bottom of the file). Whenever you change `glyberta.py` so it differs from the version committed to GitHub, increment this version number as part of the same change.

## Conventions

- All CLI subcommands share a `common` argparse parent (`--output_dir`, `--seed`, `--mlm_probability`, `--batch_size`, `--max_len`); `build_parser` wires `train`/`evaluate`/`compare` to `cmd_*` handlers via `set_defaults(func=...)`.
- Input data is always one IUPAC sequence per line; `read_sequences` strips blanks and raises on empty files.
