"""Generate a small synthetic set of glycan-like IUPAC sequences for smoke testing."""
import random

random.seed(0)

monos = ["Gal", "GlcNAc", "Man", "Neu5Ac", "Fuc", "GalNAc", "Glc", "Xyl"]
links = ["(a1-2)", "(a1-3)", "(a1-4)", "(a1-6)", "(b1-2)", "(b1-3)", "(b1-4)", "(a2-3)", "(a2-6)"]


def random_glycan():
    n = random.randint(2, 7)
    parts = [random.choice(monos)]
    for _ in range(n - 1):
        parts.append(random.choice(links))
        parts.append(random.choice(monos))
    s = "".join(parts)
    # Occasionally introduce a branch.
    if random.random() < 0.3 and len(parts) >= 5:
        branch = random.choice(monos) + random.choice(links) + random.choice(monos)
        s = s + "[" + branch + "]" + random.choice(monos)
    return s


with open("sample_sequences.txt", "w", encoding="utf-8") as fh:
    for _ in range(400):
        fh.write(random_glycan() + "\n")

print("Wrote sample_sequences.txt")
