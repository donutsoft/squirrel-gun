#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import csv
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that all images referenced in a bbox CSV exist in the positives directory."
    )
    parser.add_argument(
        "--bbox-file",
        type=Path,
        default=Path(__file__).with_name("bboxes_resolved.txt"),
        help="Path to CSV file with columns: image,label,xmin,ymin,xmax,ymax",
    )
    parser.add_argument(
        "--positives-dir",
        type=Path,
        default=Path(__file__).with_name("positives"),
        help="Directory containing the positive images referenced by the CSV",
    )
    parser.add_argument(
        "--remove-missing",
        action="store_true",
        help="If set, remove rows referencing missing images from the CSV (in-place with .bak backup)",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix appended to backup of the CSV when using --remove-missing",
    )
    parser.add_argument(
        "--list-unreferenced",
        action="store_true",
        help="List files that exist under positives_dir but are not referenced in the CSV",
    )
    parser.add_argument(
        "--unreferenced-out",
        type=Path,
        default=None,
        help="Optional path to write unreferenced file list (one per line)",
    )
    parser.add_argument(
        "--extensions",
        default="jpg,jpeg,png",
        help="Comma-separated extensions to consider when scanning positives (case-insensitive)",
    )
    parser.add_argument(
        "--move-unreferenced",
        action="store_true",
        help="Move unreferenced files to a separate directory (default: sibling 'positives_unreferenced')",
    )
    parser.add_argument(
        "--unreferenced-dest",
        type=Path,
        default=None,
        help="Destination directory for moved unreferenced files (created if missing)",
    )
    parser.add_argument(
        "--list-multibox",
        action="store_true",
        help="List images that have multiple bounding boxes (multiple CSV rows per image)",
    )
    parser.add_argument(
        "--remove-multibox",
        action="store_true",
        help="Remove all CSV rows for any image that appears multiple times (in-place with .bak backup)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bbox_file: Path = args.bbox_file
    positives_dir: Path = args.positives_dir

    if not bbox_file.exists():
        print(f"Error: bbox file not found: {bbox_file}", file=sys.stderr)
        return 2
    if not positives_dir.exists():
        print(f"Error: positives directory not found: {positives_dir}", file=sys.stderr)
        return 2

    missing = []
    seen = set()

    # First pass: scan and collect missing images, and optionally prepare filtered output
    with bbox_file.open("r", newline="") as rf:
        reader = csv.DictReader(rf)
        if "image" not in reader.fieldnames:
            print(
                f"Error: 'image' column not found in {bbox_file}. Found columns: {reader.fieldnames}",
                file=sys.stderr,
            )
            return 2

        rows = list(reader)
        fieldnames = reader.fieldnames

    # Track occurrences for multibox detection
    counts: dict[str, int] = {}
    line_numbers: dict[str, list[int]] = {}

    # Evaluate file existence per unique image and count occurrences
    for i, row in enumerate(rows, start=2):  # start=2 to account for header line as 1
        img_rel = (row.get("image") or "").strip()
        if not img_rel:
            print(f"Warning: empty image field at line {i}", file=sys.stderr)
            continue
        counts[img_rel] = counts.get(img_rel, 0) + 1
        line_numbers.setdefault(img_rel, []).append(i)
        if img_rel in seen:
            continue
        seen.add(img_rel)
        img_path = positives_dir / img_rel
        if not img_path.exists():
            missing.append(img_rel)

    total = len(seen)
    # Detect images that have multiple bbox rows in the CSV
    multibox_images = sorted([img for img, c in counts.items() if c > 1])

    # List multibox images if requested
    if args.list_multibox:
        if multibox_images:
            print("Images with multiple bounding boxes:")
            for img in multibox_images:
                nums = line_numbers.get(img, [])
                nums_str = ", ".join(map(str, nums)) if nums else "?"
                print(f" - {img} (count: {counts[img]}; lines: {nums_str})")
            print(f"\nSummary: {len(multibox_images)} images have multiple bbox rows.")
        else:
            print("No images have multiple bounding boxes.")

    # Optionally list/move files present in positives_dir not referenced by CSV
    if args.list_unreferenced or args.move_unreferenced:
        # Build set of referenced image paths (relative, as in CSV)
        referenced = set()
        for r in rows:
            name = (r.get("image") or "").strip()
            if name:
                referenced.add(name)

        # Determine which file extensions to include
        exts = {e.strip().lower().lstrip(".") for e in args.extensions.split(",") if e.strip()}

        # Scan positives_dir recursively
        all_files = []
        for p in positives_dir.rglob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower().lstrip(".")
            if exts and ext not in exts:
                continue
            rel = p.relative_to(positives_dir).as_posix()
            all_files.append(rel)

        unreferenced = sorted(set(all_files) - referenced)

        if args.list_unreferenced:
            if unreferenced:
                print("Unreferenced files in positives:")
                for f in unreferenced:
                    print(f" - {f}")
                print(f"\nSummary: {len(unreferenced)} unreferenced files (extensions: {', '.join(sorted(exts))}).")
            else:
                print("No unreferenced files in positives that match the selected extensions.")

            if args.unreferenced_out is not None:
                out_path: Path = args.unreferenced_out
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as wf:
                    for f in unreferenced:
                        wf.write(f + "\n")
                print(f"Wrote unreferenced list to: {out_path}")

        if args.move_unreferenced and unreferenced:
            # Determine destination directory
            dest_dir = args.unreferenced_dest
            if dest_dir is None:
                dest_dir = positives_dir.parent / "positives_unreferenced"

            # Safety: avoid moving into a directory inside positives_dir (would affect the scan itself)
            try:
                if dest_dir.resolve().is_relative_to(positives_dir.resolve()):
                    print(
                        f"Error: destination {dest_dir} must not be inside positives_dir {positives_dir}",
                        file=sys.stderr,
                    )
                    return 2
            except AttributeError:
                # Python < 3.9 fallback
                pos_res = str(positives_dir.resolve())
                if str(dest_dir.resolve()).startswith(pos_res + "/"):
                    print(
                        f"Error: destination {dest_dir} must not be inside positives_dir {positives_dir}",
                        file=sys.stderr,
                    )
                    return 2

            dest_dir.mkdir(parents=True, exist_ok=True)

            moved = 0
            for rel_str in unreferenced:
                src = positives_dir / rel_str
                dst = dest_dir / rel_str
                dst.parent.mkdir(parents=True, exist_ok=True)

                final_dst = dst
                if final_dst.exists():
                    # Disambiguate by adding a numeric suffix before the extension
                    stem = dst.stem
                    suffix = dst.suffix
                    parent = dst.parent
                    n = 1
                    while True:
                        candidate = parent / f"{stem} ({n}){suffix}"
                        if not candidate.exists():
                            final_dst = candidate
                            break
                        n += 1

                shutil.move(str(src), str(final_dst))
                print(f"Moved: {src.relative_to(positives_dir).as_posix()} -> {final_dst.relative_to(dest_dir).as_posix()}")
                moved += 1

            print(f"Total moved: {moved} unreferenced files to {dest_dir}")

    issues_found = False
    if missing:
        issues_found = True
        print("Missing files:")
        for m in missing:
            print(f" - {m}")
        print(f"\nSummary: {len(missing)} missing out of {total} unique images.")

    if multibox_images and not args.list_multibox and not args.remove_multibox:
        # Silent mode: still surface a brief summary so the user is aware
        print(f"Note: {len(multibox_images)} images have multiple bounding boxes. Use --list-multibox to list or --remove-multibox to drop them.")
        issues_found = True

    # Handle removal in one write if either condition applies
    to_remove_missing = set(missing) if args.remove_missing else set()
    to_remove_multibox = set(multibox_images) if args.remove_multibox else set()
    do_write = bool(to_remove_missing or to_remove_multibox)

    if do_write:
        filtered_rows = []
        removed_missing_rows = 0
        removed_multibox_rows = 0
        for r in rows:
            name = (r.get("image") or "").strip()
            if name in to_remove_missing:
                removed_missing_rows += 1
                continue
            if name in to_remove_multibox:
                removed_multibox_rows += 1
                continue
            filtered_rows.append(r)

        # Write to a temporary file, then backup and replace
        tmp_path = bbox_file.with_suffix(bbox_file.suffix + ".tmp")
        with tmp_path.open("w", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)

        backup_path = Path(str(bbox_file) + args.backup_suffix)
        try:
            # Create/overwrite backup
            backup_path.write_bytes(bbox_file.read_bytes())
            tmp_path.replace(bbox_file)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        msgs = []
        if removed_missing_rows:
            msgs.append(f"{removed_missing_rows} rows referencing missing images")
        if removed_multibox_rows:
            msgs.append(f"{removed_multibox_rows} rows from images with multiple boxes")
        if msgs:
            print(f"Removed {' and '.join(msgs)}. Backup: {backup_path}")

    if not issues_found and not multibox_images:
        print(f"All good: found all {total} unique images in '{positives_dir}', and no images have multiple boxes.")
        return 0
    else:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
