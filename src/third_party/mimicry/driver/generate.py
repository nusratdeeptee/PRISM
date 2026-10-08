"""
PRISM's own parameterized mimicry ImageNet driver.

Adapted from GRIT/baseline/mimicry/examples/run_imagenet_ubt.py +
examples/mimicry/run_experiment.py (the paper's own
run_experiments.ipynb call: dataset="Imagenet", objective="ubt",
predictor="wrn", generator="sgXL", mix_dims=(1,15)). The vendor library
under ../src/ (copied verbatim from the original repo) is untouched --
all PRISM-specific behavior (CLI args, per-task file-prefixing, PNG
export) lives in this file only, same convention as the original
project's own run_imagenet_ubt.py driver.

Must be run with THIS file living one directory level below the mimicry
root (i.e. at mimicry/driver/, next to a sibling mimicry/src/) -- see the
sys.path fixup below, needed so `from src import ...` resolves.

Model checkpoints live under PRISM/models/mimicry/ (a real copy, not a
symlink to GRIT/baseline/mimicry/models/ -- that original copy was
deleted once this copy was verified byte-identical, see chat), resolved
below via _PRISM_ROOT rather than a "../models" path relative to this
file.

Three behaviors added on top of the vendor algorithm, neither of which
touches vendor code:
  1. samples_per_class / class restriction / output naming are real CLI
     args instead of the vendor's hardcoded samples_per_class=3 / all-10
     -class loop -- needed so a SLURM array task can request a small,
     70h-safe chunk of one class instead of the whole thing at once.
  2. PNGs are written to disk THE MOMENT each sample's best candidate is
     found, not batched at the very end (2026-09-18 rewrite -- see
     _make_instrumented_append_row below). Originally this drove
     NeuralTester.test() has-run-to-completion-and-written-its-CSV as the
     ONLY point anything reached disk (test() loops all samples_per_class
     samples in memory, only calling DefaultDF.to_csv() after the full
     outer loop finishes) -- for a 334-image chunk at 150 generations/
     sample, that meant a multi-hour run with literally zero output until
     the very end, and a crash anywhere in that window (e.g. the pymoo
     AGE-MOEA ZeroDivisionError, see [[project_grit_prism_mimicry_pymoo_
     bugfix_20260918]]) lost every sample computed so far. Now each sample
     is saved concurrently with the search, matching biggan_ai_tigs'
     generate.py's own per-image save-as-you-go behavior.
  3. Confirmed from src/manipulator/_style_gan_manipulator.py's own
     _transform_image_output (full_range=False path): images are CHW
     float32 tensors in [0, 1], not normalized -- straightforward
     *255 -> uint8 -> HWC -> PIL round-trip, no un-normalization needed.

Usage:
    <mimicry-conda-env>/bin/python generate.py \\
        --class-idx 0 --num-images 480 --output-dir /path/to/mimicry_0 \\
        --file-prefix task03_
"""

import argparse
import os
import sys
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MIMICRY_ROOT = os.path.dirname(_THIS_DIR)  # driver/ -> mimicry root, where src/ lives
_PRISM_ROOT = os.path.dirname(os.path.dirname(_MIMICRY_ROOT))  # mimicry root -> src/ -> PRISM root
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _MIMICRY_ROOT)

import numpy as np
import torch
import wandb
from PIL import Image
from torchvision.models import wide_resnet50_2

from src import ExperimentConfig, NeuralTester  # noqa: E402
from src.defaults.learner_configs import PYMOO_AGE_MOEA_DEFAULT_PARAMS  # noqa: E402
from src.defaults.objective_configs import DYNAMIC_TARGETED_BOUNDARY_TESTING  # noqa: E402
from src.manipulator import StyleGANManipulator  # noqa: E402
from src.optimizer import PymooLearner  # noqa: E402
from src._neural_tester import NeuralTester as _NT  # noqa: E402
from src.persistence import DefaultDF  # noqa: E402

# Reproduces the paper's own "ubt" objective (dynamic targeted boundary
# testing) -- see run_experiment.py's OBJECTIVES dict, "ubt" entry.
OBJECTIVE = DYNAMIC_TARGETED_BOUNDARY_TESTING


# ---------------------------------------------------------------------------
# Same seed-search visibility fix as the original run_imagenet_ubt.py:
# NeuralTester._generate_seeds() does unbounded rejection sampling and its
# progress only ever reaches logging.info(), which nothing in this codebase
# configures a handler for -- so a genuinely-running job and a hung one look
# identical in a redirected stdout log. This adds flushed prints instead.
# Vendor file (src/_neural_tester.py) itself is untouched.
def _instrumented_generate_seeds(self, amount: int, cls: int):
    ws, imgs, y_hats = [], [], []
    trials = 0
    print(f"[seed-search] class={cls}: looking for {amount} valid seed(s) ...", flush=True)
    while len(ws) < amount:
        trials += 1
        if trials % 25 == 0:
            print(f"[seed-search] class={cls}: {trials} trials so far, "
                  f"{len(ws)}/{amount} found", flush=True)
        w = self._manipulator.get_w(self._get_time_seed(), cls)
        img = self._assure_rgb(self._manipulator.get_image(w))
        y_hat = self._predict(img.unsqueeze(0))
        if y_hat.argmax() == cls:
            ws.append(w)
            imgs.append(img)
            y_hats.append(y_hat)
    print(f"[seed-search] class={cls}: found {amount} valid seed(s) after {trials} trials", flush=True)
    return ws, imgs, y_hats, trials


_NT._generate_seeds = _instrumented_generate_seeds


def _tensor_to_png(nested_list, path: str) -> None:
    """CHW float32 in [0, 1] (see module docstring) -> PNG on disk."""
    arr = np.array(nested_list, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    Image.fromarray(arr).save(path)


# The vendor's own append_row, captured before any patching -- the
# instrumented replacement below still calls this so the in-memory
# DataFrame (and therefore test()'s own final CSV, harmless/still useful
# to keep) behaves exactly as before. Only the timing of the PNG export
# changes: from "after the whole samples_per_class loop finishes" to
# "the instant each sample's row is appended", i.e. the moment
# NeuralTester.test() has that sample's best candidate in hand.
_ORIGINAL_APPEND_ROW = DefaultDF.append_row


def _make_instrumented_append_row(output_dir: str, file_prefix: str):
    """Returns a DefaultDF.append_row replacement that writes that row's
    X (original) / X_prime (perturbed) images to PNG immediately, instead
    of waiting for _neural_tester.py's test() to finish its entire
    samples_per_class loop and write one big CSV at the end. Row layout is
    exactly what NeuralTester.test() builds before calling append_row:
    [X, y, X_prime, y_prime, runtime, genome] -- see that method's own
    `results = [...]` literal. Filenames/tagging match the previous
    CSV-then-export scheme's f"{file_prefix}{i}" so downstream consumers
    see identical output, just written incrementally instead of in bulk.
    """
    orig_dir = os.path.join(output_dir, "original_images")
    perturb_dir = os.path.join(output_dir, "perturbed_images")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(perturb_dir, exist_ok=True)
    counter = {"i": 0}

    def _instrumented_append_row(self, data):
        X, y, Xp, yp = data[0], data[1], data[2], data[3]
        i = counter["i"]
        counter["i"] += 1

        yp_val = yp.tolist() if hasattr(yp, "tolist") else yp
        y_prime_label = int(np.argmax(yp_val)) if isinstance(yp_val, list) else int(yp_val)
        tag = f"{file_prefix}{i}"

        _tensor_to_png(X, os.path.join(orig_dir, f"{tag}_X{y}.png"))
        _tensor_to_png(Xp, os.path.join(perturb_dir, f"{tag}_X{y}_Y{y_prime_label}.png"))
        print(f"[generate] saved sample {i} (label {y} -> {y_prime_label})", flush=True)

        _ORIGINAL_APPEND_ROW(self, data)

    _instrumented_append_row.counter = counter
    return _instrumented_append_row


def generate(class_idx: int, num_images: int, output_dir: str, file_prefix: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Install the incremental-save hook BEFORE tester.test() runs -- see
    # _make_instrumented_append_row's docstring. DefaultDF.append_row is
    # patched at the class level (same monkeypatch convention already used
    # above for NeuralTester._generate_seeds), so every row NeuralTester
    # appends during this process's test() call writes its PNGs immediately.
    instrumented_append_row = _make_instrumented_append_row(output_dir, file_prefix)
    DefaultDF.append_row = instrumented_append_row

    device = torch.device("cuda")

    sut = wide_resnet50_2(pretrained=True).to(device)
    sut.eval()

    generator_ckpt = os.path.join(_PRISM_ROOT, "models", "mimicry", "generators", "sgXL_imagenet128.pkl")
    manipulator = StyleGANManipulator(generator_ckpt, device, (1, 15), interpolate=True)

    optimizer = PymooLearner(
        **PYMOO_AGE_MOEA_DEFAULT_PARAMS,
        n_var=15 - 1,
        num_objectives=len(OBJECTIVE),
    )

    save_to = os.path.join(output_dir, f"results_mimicry_imagenet_class{class_idx}{file_prefix and '_' + file_prefix}")
    conf = ExperimentConfig(
        samples_per_class=num_images,
        generations=150,
        classes=[class_idx],
        save_to=save_to,
    )

    tester = NeuralTester(
        sut=sut,
        manipulator=manipulator,
        optimizer=optimizer,
        objectives=OBJECTIVE,
        frontier_pairs=False,
        silent_wandb=True,
        config=conf,
        restrict_classes=list(range(10)),
    )

    start = datetime.now()
    tester.test(validity_domain=False)
    elapsed = datetime.now() - start
    n_written = instrumented_append_row.counter["i"]
    print(f"[generate] class={class_idx}: tester.test() finished in {elapsed}, "
          f"{n_written} original+perturbed PNG pairs saved incrementally to {output_dir} "
          f"(plus {save_to}.csv, written by the vendor's own test() as before)", flush=True)


if __name__ == "__main__":
    wandb.login = lambda *a, **k: None  # non-interactive job, never block on wandb login
    os.environ.setdefault("WANDB_MODE", "offline")

    parser = argparse.ArgumentParser(description="PRISM mimicry ImageNet generation driver.")
    parser.add_argument("--class-idx", type=int, required=True, help="ImageNet class index (0-9).")
    parser.add_argument("--num-images", type=int, required=True, help="Number of samples to generate for this run.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write images + CSV into.")
    parser.add_argument("--file-prefix", type=str, default="", help="Prefix for output filenames (keeps parallel array tasks collision-free).")
    args = parser.parse_args()

    generate(args.class_idx, args.num_images, args.output_dir, args.file_prefix)
