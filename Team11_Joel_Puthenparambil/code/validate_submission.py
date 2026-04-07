"""
validate_submission.py
Run before submitting any result CSV:
  python validate_submission.py path/to/result.csv
"""
import pandas as pd
import sys

# load reference IDs from test set
ref = pd.read_csv("dataset_clean.csv")

# load the submission to check
sub = pd.read_csv(sys.argv[1])

# columns
assert list(sub.columns) == ["id", "answer"], \
    f"Wrong columns: {list(sub.columns)} — expected ['id', 'answer']"

# row count
assert len(sub) == len(ref), \
    f"Expected {len(ref)} rows, got {len(sub)}"

# ID order and content
assert sub["id"].tolist() == ref["id"].tolist(), \
    "IDs don't match or wrong order"

# no duplicate IDs
assert sub["id"].nunique() == len(sub), \
    "Duplicate IDs found"

# ID set matches reference
assert set(sub["id"]) == set(ref["id"]), \
    "ID set doesn't match dataset_clean.csv"

# no null answers
assert sub["answer"].notna().all(), \
    "Null/NaN answers found"

# no non-string answers
assert sub["answer"].apply(lambda x: isinstance(x, str)).all(), \
    "Non-string answers found"

# no blank/whitespace-only answers
assert (sub["answer"].str.strip() != "").all(), \
    "Blank or whitespace-only answers found"

# catch crashed model writing the same fallback repeatedly
top_freq = sub["answer"].value_counts().iloc[0]
assert top_freq < 20, \
    f"Suspiciously repeated answer ({top_freq}x) — model may have crashed"

print(f"OK — {sys.argv[1]} looks valid ({len(sub)} rows)")
