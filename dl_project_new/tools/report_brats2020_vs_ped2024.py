import argparse
import json
from pathlib import Path


DEFAULT_2020 = Path("C:/Users/arpana/Downloads/BrainSegNet/outputs/all_models_eval_results_brats2020.json")
DEFAULT_2024 = Path("C:/Users/arpana/Downloads/BrainSegNet/outputs/all_models_eval_results_brats_ped2024.json")
METRICS = ("WT", "TC", "ET")
MODELS = ("Baseline1", "Baseline2", "BrainSegNet")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def f3(x):
    return f"{x:.3f}"


def fd(x):
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.3f}"


def print_table(title, headers, rows):
    print(f"\n{title}")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    print("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    print(sep)


def build_summary_rows(r2020, r2024):
    rows = []
    for model in MODELS:
        m20 = r2020[model]["MEAN"]
        m24 = r2024[model]["MEAN"]
        row = [model]
        for metric in METRICS:
            row.extend([f3(m20[metric]), f3(m24[metric]), fd(m24[metric] - m20[metric])])
        rows.append(row)
    return rows


def build_setting_rows(r2020, r2024, setting):
    rows = []
    for model in MODELS:
        s20 = r2020[model][setting]
        s24 = r2024[model][setting]
        row = [model]
        for metric in METRICS:
            row.extend([f3(s20[metric]), f3(s24[metric]), fd(s24[metric] - s20[metric])])
        rows.append(row)
    return rows


def build_full_rows(r2020, r2024, model):
    settings = [k for k in r2020[model].keys() if k != "MEAN"]
    rows = []
    for setting in settings:
        s20 = r2020[model][setting]
        s24 = r2024[model][setting]
        rows.append(
            [
                setting,
                f3(s20["WT"]),
                f3(s24["WT"]),
                fd(s24["WT"] - s20["WT"]),
                f3(s20["TC"]),
                f3(s24["TC"]),
                fd(s24["TC"] - s20["TC"]),
                f3(s20["ET"]),
                f3(s24["ET"]),
                fd(s24["ET"] - s20["ET"]),
            ]
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Print side-by-side BrainSegNet vs baselines comparison for BraTS2020 and BraTS PED2024."
    )
    parser.add_argument("--results-2020", type=Path, default=DEFAULT_2020, help="Path to BraTS2020 results JSON.")
    parser.add_argument("--results-2024", type=Path, default=DEFAULT_2024, help="Path to BraTS PED2024 results JSON.")
    args = parser.parse_args()

    r2020 = load_json(args.results_2020)
    r2024 = load_json(args.results_2024)

    headers = [
        "Model",
        "WT@2020",
        "WT@2024",
        "WT Delta",
        "TC@2020",
        "TC@2024",
        "TC Delta",
        "ET@2020",
        "ET@2024",
        "ET Delta",
    ]

    print(f"Loaded 2020: {args.results_2020}")
    print(f"Loaded 2024: {args.results_2024}")
    print("\nDelta is (2024 - 2020). Negative means performance drop on PED2024.")

    print_table("Table 1: Mean over all 15 modality settings", headers, build_summary_rows(r2020, r2024))
    print_table(
        "Table 2: All 4 modalities only",
        headers,
        build_setting_rows(r2020, r2024, setting="All 4 modalities"),
    )

    per_model_headers = [
        "Modality Setting",
        "WT@2020",
        "WT@2024",
        "WT Delta",
        "TC@2020",
        "TC@2024",
        "TC Delta",
        "ET@2020",
        "ET@2024",
        "ET Delta",
    ]
    for model in MODELS:
        print_table(f"Table 3 ({model}): Full modality-wise comparison", per_model_headers, build_full_rows(r2020, r2024, model))


if __name__ == "__main__":
    main()
