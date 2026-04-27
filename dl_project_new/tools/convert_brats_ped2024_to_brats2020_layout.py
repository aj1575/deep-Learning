import os
import argparse
import shutil
from pathlib import Path


MODALITY_MAP = {
    "t1n": "t1",
    "t1c": "t1ce",
    "t2w": "t2",
    "t2f": "flair",
}


def pick_nii_file(path: Path) -> Path:
    if path.is_file() and path.suffix in {".nii", ".gz"}:
        return path
    if path.is_dir():
        files = [p for p in path.iterdir() if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))]
        if len(files) == 1:
            return files[0]
        if len(files) > 1:
            # Prefer names that look like image volumes.
            files_sorted = sorted(files, key=lambda p: p.name)
            return files_sorted[0]
    raise FileNotFoundError(f"No NIfTI file found in {path}")


def find_source_file(case_dir: Path, case_id: str, src_mod: str) -> Path:
    direct_file = case_dir / f"{case_id}-{src_mod}.nii"
    if direct_file.is_file():
        return direct_file
    if direct_file.is_dir():
        return pick_nii_file(direct_file)
    direct_file_gz = case_dir / f"{case_id}-{src_mod}.nii.gz"
    if direct_file_gz.is_file():
        return direct_file_gz
    if direct_file_gz.is_dir():
        return pick_nii_file(direct_file_gz)

    wrapped_dir = case_dir / f"{case_id}-{src_mod}.nii"
    if wrapped_dir.exists():
        return pick_nii_file(wrapped_dir)

    wrapped_dir_gz = case_dir / f"{case_id}-{src_mod}.nii.gz"
    if wrapped_dir_gz.exists():
        return pick_nii_file(wrapped_dir_gz)

    raise FileNotFoundError(f"Could not locate {src_mod} for case {case_id} under {case_dir}")


def convert_case(case_dir: Path, out_root: Path) -> bool:
    case_id = case_dir.name
    out_case = out_root / case_id
    out_case.mkdir(parents=True, exist_ok=True)

    try:
        for src_mod, dst_mod in MODALITY_MAP.items():
            src = find_source_file(case_dir, case_id, src_mod)
            dst = out_case / f"{case_id}_{dst_mod}.nii"
            shutil.copy2(src, dst)

        seg_src = find_source_file(case_dir, case_id, "seg")
        seg_dst = out_case / f"{case_id}_seg.nii"
        shutil.copy2(seg_src, seg_dst)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Skipping case {case_id}: {exc}")
        return False


def get_args():
    p = argparse.ArgumentParser(
        description="Convert BraTS PED 2024 Kaggle layout to BraTS2020-style case folder layout."
    )
    p.add_argument("--input-root", required=True, help="Path to BraTS-PEDs2024_Training folder.")
    p.add_argument("--output-root", required=True, help="Output root for converted case folders.")
    return p.parse_args()


def main():
    args = get_args()
    in_root = Path(args.input_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = [p for p in in_root.iterdir() if p.is_dir() and p.name.startswith("BraTS-PED-")]
    case_dirs.sort()
    print(f"Found candidate cases: {len(case_dirs)}")

    ok = 0
    for case_dir in case_dirs:
        if convert_case(case_dir, out_root):
            ok += 1

    print(f"Converted cases: {ok}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()
