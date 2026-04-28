import json
from pathlib import Path


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def f3(x):
    return f"{x:.3f}"


def delta(a, b):
    d = b - a
    return f"{d:+.3f}"


def row(model, b20, b24, a20, a24):
    return (
        f"{model:12s} | "
        f"{f3(b20['WT'])}/{f3(b20['TC'])}/{f3(b20['ET'])} | "
        f"{f3(b24['WT'])}/{f3(b24['TC'])}/{f3(b24['ET'])} | "
        f"{f3(a20['WT'])}/{f3(a20['TC'])}/{f3(a20['ET'])} | "
        f"{f3(a24['WT'])}/{f3(a24['TC'])}/{f3(a24['ET'])} | "
        f"{delta(b24['WT'], a24['WT'])}/{delta(b24['TC'], a24['TC'])}/{delta(b24['ET'], a24['ET'])}"
    )


def main():
    before20 = load("C:/Users/arpana/Downloads/BrainSegNet/outputs/all_models_eval_results_brats2020.json")
    before24 = load("C:/Users/arpana/Downloads/BrainSegNet/outputs/all_models_eval_results_brats_ped2024.json")
    after20 = load(
        "C:/Users/arpana/Downloads/BrainSegNet_backup_outputs/ped2024_finetune/"
        "all_models_eval_results_brats2020_with_ped2024_finetuned_brainsegnet.json"
    )
    after24 = load(
        "C:/Users/arpana/Downloads/BrainSegNet_backup_outputs/ped2024_finetune/"
        "all_models_eval_results_ped2024_finetuned_brainsegnet.json"
    )

    lines = []
    lines.append("Before vs After PED2024 Fine-tuning (MEAN over 15 modality settings)")
    lines.append("Metrics format: WT/TC/ET")
    lines.append("")
    lines.append(
        "Model        | Before@2020 | Before@2024 | After@2020 | After@2024 | Gain on 2024 (After-Before)"
    )
    lines.append("-" * 120)

    for model in ("Baseline1", "Baseline2", "BrainSegNet"):
        lines.append(
            row(
                model,
                before20[model]["MEAN"],
                before24[model]["MEAN"],
                after20[model]["MEAN"],
                after24[model]["MEAN"],
            )
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- Baselines were not fine-tuned; tiny differences across runs come from sampling/crop stochasticity.")
    lines.append("- BrainSegNet fine-tuning on PED2024 massively improves PED2024 score but reduces BraTS2020 score (catastrophic forgetting).")

    out_path = Path(
        "C:/Users/arpana/Downloads/BrainSegNet_backup_outputs/ped2024_finetune/before_after_finetune_table.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report -> {out_path}")


if __name__ == "__main__":
    main()
