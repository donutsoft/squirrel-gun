#!/usr/bin/env bash
set -euo pipefail

remote_host="pi@192.168.1.155"
remote_data_dir=/home/pi/squirrel-training-data
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
local_dataset_dir="$script_dir/dataset"
local_negatives_dir="$local_dataset_dir/negatives"
local_positives_dir="$local_dataset_dir/positives"
local_bbox_file="$local_dataset_dir/bboxes.txt"

pull_tmp=$(mktemp -d "${TMPDIR:-/tmp}/squirrel-training-pull.XXXXXX")
bbox_tmp=""

cleanup() {
  if [ -n "$bbox_tmp" ] && [ -f "$bbox_tmp" ]; then
    rm -f "$bbox_tmp"
  fi
  rm -rf "$pull_tmp"
}
trap cleanup EXIT

echo "Pulling training data from $remote_host:$remote_data_dir"
scp -r "$remote_host:$remote_data_dir" "$pull_tmp/"

pulled_data_dir="$pull_tmp/squirrel-training-data"
pulled_negatives_dir="$pulled_data_dir/negatives"
pulled_positives_dir="$pulled_data_dir/positives"
pulled_bbox_file="$pulled_data_dir/bboxes.txt"

if [ ! -d "$pulled_negatives_dir" ]; then
  echo "Missing remote negatives directory: $pulled_negatives_dir" >&2
  exit 1
fi
if [ ! -d "$pulled_positives_dir" ]; then
  echo "Missing remote positives directory: $pulled_positives_dir" >&2
  exit 1
fi
if [ ! -f "$pulled_bbox_file" ]; then
  echo "Missing remote bbox file: $pulled_bbox_file" >&2
  exit 1
fi

mkdir -p "$local_negatives_dir" "$local_positives_dir"

# Copy only image files. Existing same-named mined frames are updated in place;
# unrelated local training images are left untouched.
find "$pulled_negatives_dir" -maxdepth 1 -type f -name '*.jpg' \
  -exec cp -p {} "$local_negatives_dir/" \;
find "$pulled_positives_dir" -maxdepth 1 -type f -name '*.jpg' \
  -exec cp -p {} "$local_positives_dir/" \;

negative_count=$(find "$pulled_negatives_dir" -maxdepth 1 -type f -name '*.jpg' | wc -l | tr -d ' ')
positive_count=$(find "$pulled_positives_dir" -maxdepth 1 -type f -name '*.jpg' | wc -l | tr -d ' ')

# Replace local bbox rows for images present in the Pi dataset, then append the
# current remote rows. This makes repeated pulls idempotent while preserving
# annotations for all unrelated local images.
if [ -f "$local_bbox_file" ]; then
  bbox_tmp=$(mktemp "$local_dataset_dir/.bboxes.pull.XXXXXX")
  awk -F, '
    NR == FNR {
      if (FNR > 1) remote_images[$1] = 1
      next
    }
    FNR == 1 { print; next }
    !($1 in remote_images) { print }
  ' "$pulled_bbox_file" "$local_bbox_file" > "$bbox_tmp"
  tail -n +2 "$pulled_bbox_file" >> "$bbox_tmp"
  mv "$bbox_tmp" "$local_bbox_file"
  bbox_tmp=""
else
  cp -p "$pulled_bbox_file" "$local_bbox_file"
fi

bbox_count=$(tail -n +2 "$pulled_bbox_file" | wc -l | tr -d ' ')

echo "Pulled $negative_count false-positive frame(s) into $local_negatives_dir"
echo "Pulled $positive_count true-positive frame(s) into $local_positives_dir"
echo "Merged $bbox_count bounding-box row(s) into $local_bbox_file"
