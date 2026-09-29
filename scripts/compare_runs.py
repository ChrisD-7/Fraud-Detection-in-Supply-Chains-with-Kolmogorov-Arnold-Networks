"""Compare two run outputs key by key: are identical configs bitwise reproducible?"""
import json
import sys
from pathlib import Path

KEYS = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]

a_path, b_path = sys.argv[1], sys.argv[2]
a = json.loads(Path(a_path).read_text())
b = json.loads(Path(b_path).read_text())

print(f"{'key':<28}{'A':>10}{'B':>10}{'delta':>12}")
worst = 0.0
for section in ("test", "train"):
    for key in KEYS:
        va, vb = a[section][key], b[section][key]
        delta = abs(va - vb)
        worst = max(worst, delta)
        flag = "" if delta == 0 else "   <-- differs"
        print(f"{section + '.' + key:<28}{va:>10.6f}{vb:>10.6f}{delta:>12.2e}{flag}")

for key, path_a, path_b in [
    ("single_feature_baseline.best_roc_auc",
     a["single_feature_baseline"]["best_roc_auc"], b["single_feature_baseline"]["best_roc_auc"]),
    ("single_feature_baseline.best_feature",
     a["single_feature_baseline"]["best_feature"], b["single_feature_baseline"]["best_feature"]),
    ("single_feature_baseline.at_chance",
     a["single_feature_baseline"]["at_chance"], b["single_feature_baseline"]["at_chance"]),
    ("test.confusion", str(a["test"]["confusion"]), str(b["test"]["confusion"])),
    ("formula_test_accuracy",
     a.get("symbolic", {}).get("formula_test_accuracy"),
     b.get("symbolic", {}).get("formula_test_accuracy")),
]:
    same = "same" if path_a == path_b else "DIFFERS"
    print(f"{key:<28}{str(path_a):>28} | {str(path_b):<28} {same}")

print()
print(f"worst metric delta: {worst:.2e}")
print("REPRODUCIBLE" if worst == 0.0 else "NOT BITWISE REPRODUCIBLE")
