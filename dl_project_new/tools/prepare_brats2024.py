import os
import argparse
import zipfile
from pathlib import Path


REQUIRED_SUFFIXES = ["_t1", "_t1ce", "_t2", "_flair"]
OPTIONAL_SUFFIX = "_seg"


def is_nii(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def case_has_modalities(case_dir: Path) -> bool:
    names = [p.name.lower() for p in case_dir.iterdir() if p.is_file() and is_nii(p)]
    return all(any(sfx in n for n in names) for sfx in REQUIRED_SUFFIXES)


def discover_case_root(base_dir: Path) -> Path:
    # Prefer a directory where direct children are case folders with BraTS-like files.
    candidates = [base_dir] + [p for p in base_dir.rglob("*") if p.is_dir()]
    best = None
    best_count = 0
    for cand in candidates:
        case_dirs = [d for d in cand.iterdir() if d.is_dir()] if cand.exists() else []
        valid = sum(1 for d in case_dirs if case_has_modalities(d))
        if valid > best_count:
            best_count = valid
            best = cand
    if best is None or best_count == 0:
        raise RuntimeError(
            f"Could not find a BraTS-style case root under: {base_dir}\n"
            "Expected case folders each containing t1/t1ce/t2/flair NIfTI files."
        )
    return best


def maybe_extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def summarize(case_root: Path) -> None:
    case_dirs = [d for d in case_root.iterdir() if d.is_dir()]
    valid = [d for d in case_dirs if case_has_modalities(d)]
    with_seg = 0
    for d in valid:
        names = [p.name.lower() for p in d.iterdir() if p.is_file() and is_nii(p)]
        if any(OPTIONAL_SUFFIX in n for n in names):
            with_seg += 1
    print(f"Discovered case root: {case_root}")
    print(f"Valid cases: {len(valid)}")
    print(f"Cases with segmentations ({OPTIONAL_SUFFIX}): {with_seg}")
    print("Use this as --data-root for evaluation.")


def get_args():
    p = argparse.ArgumentParser(
        description="Extract and locate BraTS 2024 dataset root for evaluation."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to BraTS2024 zip file or already-extracted directory.",
    )
    p.add_argument(
        "--extract-to",
        default=None,
        help="Output directory for zip extraction. Defaults to sibling folder of zip.",
    )
    return p.parse_args()


def main():
    args = get_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".zip":
            raise ValueError("If --input is a file, it must be a .zip archive.")
        extract_to = Path(args.extract_to).expanduser().resolve() if args.extract_to else (
            input_path.parent / f"{input_path.stem}_extracted"
        )
        extracted_dir = maybe_extract_zip(input_path, extract_to)
        case_root = discover_case_root(extracted_dir)
    else:
        case_root = discover_case_root(input_path)

    summarize(case_root)


if __name__ == "__main__":
    main()
