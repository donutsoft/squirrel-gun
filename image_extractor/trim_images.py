#!/usr/bin/env python3
"""
Randomly delete images in a directory until only the desired number remain.

Usage examples:
  python3 trim_images.py /path/to/stills --target 500 --yes
  python3 trim_images.py ./stills --target 1000 --dry-run --ext jpg jpeg

Features:
  - Supports common image extensions (case-insensitive)
  - Optional recursive search
  - Deterministic sampling with --seed
  - Safety by default: prompts for confirmation unless --yes, supports --dry-run
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


DEFAULT_EXTS = [
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "gif",
    "webp",
]


def gather_images(root: Path, exts: list[str], recursive: bool) -> list[Path]:
    exts = [e.lower().lstrip(".") for e in exts]
    results: list[Path] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                results.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                results.append(p)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        help="Directory containing images to trim",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=int,
        required=True,
        help="Desired number of images to keep",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=DEFAULT_EXTS,
        help="Image extensions to consider (default: %(default)s)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recurse into subdirectories",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic selection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Do not prompt for confirmation; proceed with deletion",
    )

    args = parser.parse_args(argv)

    root = Path(args.directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Directory not found or not a directory: {root}", file=sys.stderr)
        return 2
    if args.target < 0:
        print("--target must be non-negative", file=sys.stderr)
        return 2

    files = gather_images(root, args.ext, args.recursive)
    total = len(files)
    print(f"Found {total} images in {root}")

    if total <= args.target:
        print(f"No deletion needed: total {total} <= target {args.target}")
        return 0

    to_delete_count = total - args.target
    rng = random.Random(args.seed)
    # Shuffle and take first N to delete for determinism when seed provided
    rng.shuffle(files)
    delete_list = files[:to_delete_count]

    print(f"Preparing to delete {to_delete_count} image(s) to reach target {args.target}.")
    if args.dry_run:
        print("Dry-run: listing files that would be deleted:")
        for p in delete_list:
            # Show paths relative to the root for readability
            try:
                rel = p.relative_to(root)
            except Exception:
                rel = p
            print(f"  {rel}")
        print("Dry-run complete. No files were deleted.")
        return 0

    if not args.yes:
        confirm = input(
            f"Delete {to_delete_count} files from '{root}'? Type 'yes' to confirm: "
        ).strip()
        if confirm.lower() != "yes":
            print("Aborted by user. No files were deleted.")
            return 1

    deleted = 0
    errors = 0
    for p in delete_list:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {p}: {e}", file=sys.stderr)
            errors += 1

    print(f"Deleted {deleted} file(s). {total - deleted} remain.")
    if errors:
        print(f"Encountered {errors} error(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

