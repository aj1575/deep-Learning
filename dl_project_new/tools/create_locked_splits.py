import os
import sys
import json
import argparse
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import find_valid_patients


def get_args():
    p = argparse.ArgumentParser(description="Create fixed train/val/test patient splits for fair model comparison.")
    p.add_argument("--data-root", required=True, help="Dataset root containing BraTS case folders.")
    p.add_argument("--train", type=int, default=219, help="Train split size.")
    p.add_argument("--val", type=int, default=50, help="Validation split size.")
    p.add_argument("--test", type=int, default=99, help="Test split size.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed used once to create fixed split.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    return p.parse_args()


def main():
    args = get_args()
    ids = find_valid_patients(args.data_root)
    need = args.train + args.val + args.test
    if len(ids) < need:
        raise ValueError(f"Not enough valid patients. Need {need}, found {len(ids)}")

    random.seed(args.seed)
    random.shuffle(ids)

    train_ids = ids[:args.train]
    val_ids = ids[args.train:args.train + args.val]
    test_ids = ids[args.train + args.val:need]

    payload = {
        "meta": {
            "data_root": args.data_root,
            "seed": args.seed,
            "counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        },
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved locked splits -> {args.out}")
    print(f"Counts: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")


if __name__ == "__main__":
    main()
