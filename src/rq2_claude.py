#!/usr/bin/env python3
"""
RQ2 Stage-2 open-coding runner.

For every (image_folder -> feature_list) entry in a mapping JSON, this script:
  1. lists every image in that local folder,
  2. sends each image + the coding prompt to Claude Sonnet 5 via the Anthropic API,
  3. parses the model's per-feature judgments (is_present / has_attention),
  4. writes one CSV per folder into results/rq2/.

Crash-safe / resumable: the CSV for each folder is rewritten to disk after every
single image (not just after the folder finishes). If the script is stopped and
re-run, it re-reads each folder's existing CSV, skips any image that already has
a complete row (no blank fields), and only calls the API for images that are
missing or previously failed -- so a rerun never re-pays for already-coded images.

Install once:
    pip install anthropic

Set your API key (never hardcode it in this file), then run from the PRISM root:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python src/rq2_claude.py
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time

import anthropic

# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------

# Reads from the ANTHROPIC_API_KEY environment variable -- never hardcode a
# key in this file. Set it before running:
#   export ANTHROPIC_API_KEY="sk-ant-..."
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

MODEL = "claude-sonnet-5"

# Path to the folder->features mapping JSON you attached.
MAPPING_PATH = "results/rq2_75_img_fet_mapping.json"

# Where per-folder CSVs get written.
OUTPUT_DIR = "results/rq2"

# If your image folder paths in the mapping JSON are relative to something
# other than the current working directory, set this prefix. Leave "" if
# the paths in the JSON are already correct relative (or absolute) paths.
BASE_DIR = ""

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Small delay between API calls to stay comfortably under rate limits.
REQUEST_DELAY_SECONDS = 0.4

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5

# Claude Sonnet 5 does extended thinking by default, and thinking tokens
# count against max_tokens. A too-small budget (500 was not enough) can let
# thinking consume the whole cap before any answer text is emitted, causing
# stop_reason="max_tokens" with zero text content -- and since the same
# image reliably triggers similar-length reasoning, this fails identically
# on every retry, not just transiently. Billing is by tokens actually used,
# not this cap, so a generous ceiling costs nothing on the common case.
MAX_OUTPUT_TOKENS = 4096

# ----------------------------------------------------------------------
# Prompt (single-image variant of the Stage-2 open-coding protocol)
# ----------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a careful visual annotator performing open coding for a research \
study on deep neural network attention. You will inspect ONE image panel and, for a given list of \
semantic features, judge whether each feature is (a) present in the scene and (b) receiving \
noticeable model attention. Code this image independently and spontaneously, based only on what is \
visible in it — do not assume patterns from any other image.

INPUT FORMAT
The image is a single panel split down the middle into two halves of equal width:
- Left half: the original RGB scene, unmodified.
- Right half: the same scene overlaid with a model attention/saliency heatmap, where color encodes \
attention intensity on a low-to-high scale, rendered blue (low) -> cyan/green (medium) -> \
yellow/orange/red (high).

DEFINITIONS (apply exactly, do not reinterpret)

Is present (1/0):
- 1 if the feature is visibly identifiable anywhere in the LEFT (original) panel, even partially, \
small, or partially occluded.
- 0 if the feature cannot be identified in the scene at all.
- Judge presence from the original image only, never from the heatmap.

Having attention (1/0):
- 1 if the region of the RIGHT (heatmap) panel directly over that feature shows a noticeable amount \
of non-blue color -- i.e., a clearly visible patch/cluster of cyan, green, yellow, orange, or red, OR \
a visibly elevated density of non-blue speckle compared to the surrounding/background regions. It \
does not need to be a solid contiguous blob.
- 0 if the region over the feature is essentially pure/near-pure blue, or shows only the same sparse, \
low-density speckle that appears uniformly across unrelated background regions (sky, empty pavement, \
foliage) elsewhere in the same image. This ambient low-level scatter is noise, not attention.
- If is_present = 0, always set has_attention = 0.
- Always judge relative to this image's own background noise floor, not "is there any color at all."

PROCEDURE
1. Split the image into left (original) and right (heatmap) halves.
2. Scan a few clearly irrelevant background regions in the heatmap half (sky, empty road, foliage) \
to establish this image's noise baseline.
3. For each feature below, in order: locate it in the left panel. If not found, is_present=0 and \
has_attention=0. If found, locate the same region in the right panel, compare it against the \
baseline from step 2, and set has_attention=1 only if that region is noticeably warmer/denser than \
the baseline.

FEATURE LIST (use these exact strings as JSON keys, in this order): {feature_list}

OUTPUT FORMAT
Respond with ONLY a single JSON object and nothing else -- no markdown code fences, no explanation, \
no extra text before or after it. The JSON object must have exactly one key per feature listed above, \
each mapping to an object with integer 0/1 values, in this exact shape:

{{"feature name": {{"is_present": 0, "has_attention": 0}}, "another feature": {{"is_present": 1, "has_attention": 1}}}}
"""

USER_TEXT_TEMPLATE = "Image filename: {filename}\nCode this image now, following the instructions exactly."


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def natural_sort_key(name):
    """Sort filenames like 9.png, 48.png, 130.png in numeric order."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def list_images(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")
    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    files.sort(key=natural_sort_key)
    return files


def media_type_for(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    raise ValueError(f"Unsupported image extension: {ext}")


def csv_name_for_folder(folder_key):
    """
    Turn a mapping key like 'rq2_data/mimicry_5/idg/image_75' into a
    short, readable csv filename like 'mimicry_5_idg.csv'.

    The trailing 'image_<N>' segment is dropped from the name entirely when
    N == 75 (the original/default image count, so existing CSVs keep their
    established names and nothing already on disk gets clobbered). For any
    other N (e.g. the 25-image subset), '_<N>' is appended instead, so e.g.
    'rq2_data/s1/idg/image_25' -> 's1_idg_25.csv' -- a distinct file from
    's1_idg.csv' rather than overwriting it.
    """
    parts = [p for p in folder_key.strip("/").split("/") if p]
    suffix = ""
    if parts and parts[-1].lower().startswith("image"):
        match = re.search(r"(\d+)$", parts[-1])
        if match and match.group(1) != "75":
            suffix = f"_{match.group(1)}"
        parts = parts[:-1]
    if parts and parts[0] == "rq2_data":
        parts = parts[1:]
    if not parts:
        parts = ["output"]
    return "_".join(parts) + suffix + ".csv"


def extract_json_object(text):
    """
    Pull a JSON object out of the model's reply even if it wrapped it in
    markdown fences or added stray whitespace/text around it.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Fall back to the first {...} block found.
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    return json.loads(text)


def code_one_image(client, image_path, filename, feature_list):
    with open(image_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(feature_list=json.dumps(feature_list))
    user_text = USER_TEXT_TEMPLATE.format(filename=filename)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type_for(filename),
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
            )
            raw_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            parsed = extract_json_object(raw_text)
            return parsed
        except Exception as e:  # noqa: BLE001 - we want to retry on any failure
            last_error = e
            print(f"    ! attempt {attempt}/{MAX_RETRIES} failed for {filename}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(f"    !! giving up on {filename} after {MAX_RETRIES} attempts: {last_error}")
    return None


def row_for_image(filename, feature_list, parsed):
    row = [filename]
    for feature in feature_list:
        if parsed is None or feature not in parsed:
            # Model failed / didn't return this feature -- leave blank rather
            # than guessing, so failures are easy to spot and re-run later.
            row.extend(["", ""])
            continue
        entry = parsed[feature]
        row.append(entry.get("is_present", ""))
        row.append(entry.get("has_attention", ""))
    return row


def row_is_complete(row, feature_list):
    expected_len = 1 + 2 * len(feature_list)
    if len(row) != expected_len:
        return False
    return all(value != "" for value in row[1:])


def load_existing_results(out_path, header):
    """
    Return {filename: row} for images already fully coded in out_path.

    Rows left over from a previous failed attempt (blank fields) are dropped
    so those images get retried. If the file's header doesn't match the
    current feature list (e.g. the mapping JSON changed), the stale file is
    backed up rather than silently overwritten.
    """
    if not os.path.exists(out_path):
        return {}
    with open(out_path, newline="") as f:
        reader = csv.reader(f)
        try:
            existing_header = next(reader)
        except StopIteration:
            return {}
        if existing_header != header:
            backup_path = out_path + ".stale"
            print(f"    ! {out_path} header doesn't match the current feature "
                  f"list -- backing up to {backup_path} and starting fresh")
            os.replace(out_path, backup_path)
            return {}
        rows = {row[0]: row for row in reader if row}
    return rows


def write_csv_snapshot(out_path, header, images, results):
    """Atomically (re)write the folder's CSV with whatever has been coded so far."""
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for filename in images:
            if filename in results:
                writer.writerow(results[filename])
    os.replace(tmp_path, out_path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RQ2 stage-2 open-coding runner")
    parser.add_argument(
        "--mapping-path", default=MAPPING_PATH,
        help=f"Path to the folder->features mapping JSON (default: {MAPPING_PATH}).",
    )
    parser.add_argument(
        "--folders", default=None,
        help="Comma-separated subset of mapping-JSON keys to process this run "
             "(e.g. for splitting work across parallel SLURM jobs). Default: "
             "process every folder in the mapping.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not API_KEY:
        print("Set the ANTHROPIC_API_KEY environment variable before running.")
        sys.exit(1)

    with open(args.mapping_path, "r") as f:
        mapping = json.load(f)

    if args.folders:
        requested = [key.strip() for key in args.folders.split(",") if key.strip()]
        missing = [key for key in requested if key not in mapping]
        if missing:
            print(f"Unknown folder key(s) not in {args.mapping_path}: {missing}")
            sys.exit(1)
        mapping = {key: mapping[key] for key in requested}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = anthropic.Anthropic(api_key=API_KEY)

    for folder_key, feature_list in mapping.items():
        folder_path = os.path.join(BASE_DIR, folder_key) if BASE_DIR else folder_key
        out_name = csv_name_for_folder(folder_key)
        out_path = os.path.join(OUTPUT_DIR, out_name)

        print(f"\n=== Folder: {folder_key} -> {out_path} ===")
        images = list_images(folder_path)
        print(f"    {len(images)} images, {len(feature_list)} features")

        header = ["Image name"]
        for feature in feature_list:
            header.append(f"{feature} is present")
            header.append(f"{feature} has attention")

        results = load_existing_results(out_path, header)
        results = {
            filename: row for filename, row in results.items()
            if row_is_complete(row, feature_list)
        }
        pending = [filename for filename in images if filename not in results]
        print(f"    {len(images)} images total, {len(results)} already coded, "
              f"{len(pending)} remaining")

        if not pending:
            print(f"    already complete -> {out_path}")
            continue

        # Persist immediately so a crash before the first image still leaves
        # a valid header + any already-coded rows on disk.
        write_csv_snapshot(out_path, header, images, results)

        for i, filename in enumerate(pending, start=1):
            image_path = os.path.join(folder_path, filename)
            print(f"    [{i}/{len(pending)}] {filename}")
            parsed = code_one_image(client, image_path, filename, feature_list)
            results[filename] = row_for_image(filename, feature_list, parsed)
            write_csv_snapshot(out_path, header, images, results)
            time.sleep(REQUEST_DELAY_SECONDS)

        print(f"    done -> {out_path}")

    print("\nAll folders processed.")


if __name__ == "__main__":
    main()