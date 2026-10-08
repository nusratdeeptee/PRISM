# PRISM

This repository accompanies the paper “PRISM: Attribution-Guided Discovery of Human-Interpretable Semantic Modes in DNN Testing” and contains the implementation and artifacts. PRISM enables attention-based failure characterization for image-classifying models under
test: given a set of pass/fail-labeled test images, finds which visible objects the model's attention correlates with failure.

## SGSM Requirements

<table border="1" cellpadding="6" style="border-collapse:collapse">
<tr><td width="60" align="center"><strong>Id</strong></td><td width="500" align="center"><strong>Precondition</strong></td><td width="220" align="center"><strong>Postcondition</strong></td></tr>
<tr><td align="center">S1</td><td align="center">A vehicle is within 10 meters, in front, and in the same lane</td><td align="center">not accelerate</td></tr>
<tr><td align="center">S2</td><td align="center">The ego lane is controlled by a red or yellow light</td><td align="center">decelerate</td></tr>
<tr><td align="center">S3</td><td align="center">The ego lane is controlled by a green light, and no vehicle is in front, in the same lane, and within 10 meters</td><td align="center">accelerate</td></tr>
<tr><td align="center">S4</td><td align="center">The ego is in the rightmost lane and is not in an intersection</td><td align="center">do not steer to the right</td></tr>
<tr><td align="center">S5</td><td align="center">The ego is in the leftmost lane and is not in an intersection</td><td align="center">do not steer to the left</td></tr>
<tr><td align="center">S6</td><td align="center">A vehicle is in the lane to the left and within 7 meters</td><td align="center">do not steer to the left</td></tr>
<tr><td align="center">S7</td><td align="center">A vehicle is in the lane to the right and within 7 meters</td><td align="center">do not steer to the right</td></tr>
</table>

## Getting Started

### Environment

All of our experiments were conducted using Python 3.11.13, CUDA 12.8, and
NVIDIA A100 and A6000 GPUs.

After creating a python environment, install torch/torchvision from the CUDA
12.8 wheel index, then the remaining dependencies:

<code>pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128</code>

<code>pip install -r requirements.txt</code>

`requirements.txt` covers `src/main.py` and everything it drives
(`generate_segments.py`, `ig2.py`, `idg.py`, `generate_analysis.py`,
`src/common/`, `rq*.py`, `src/paper_result_related_code/`). It does not cover
`src/third_party/` -- those are vendored baselines (B2T, EP-DNN-testing,
Mimicry, GenAI-TIG) used for the RQ3 model-repair comparison and the ImageNet
test-generation baselines, each with its own, separate dependency set; see the
corresponding subfolder for its own setup.

`nltk`'s wordnet corpus and the Hugging Face model cache are expected under
`~/nltk_data` and `~/.cache/huggingface` respectively -- if those are empty,
`nltk.download('wordnet')` and a first model load will populate them.

## Pipeline

1. **`src/generate_segments.py`** -- Qwen2.5-VL names every visible object in
   each image, Grounding DINO grounds those names into boxes, SAM2 turns each
   box into a mask. Runs once per dataset, independent of attention method.
2. **`src/ig2.py`** / **`src/idg.py`** -- each computes one attention map per
   image (IG2 or IDG), independent of segmentation and of each other.
3. **`src/generate_analysis.py`** -- combines segments + attention: per
   object, tests whether its attention is statistically above chance
   (permutation test); across the dataset, canonicalizes labels, tests which
   labels' presence/attention share differs between pass and fail (bootstrap
   CI + Benjamini-Hochberg FDR correction), fits a Random Forest + SHAP over
   all labels jointly to check for a learnable feature *combination*, and
   writes one `analysis_report.md` per dataset.

Steps 1-3 can run as independent jobs -- segmentation and both attention
methods are mutually independent, so all three can run in parallel; analysis
waits on whichever of segments/attention it needs.

4. **`src/main.py`** -- ties the three steps together as PRISM's single entry
   point, one `--mode` per step (plus two modes that chain all three in one
   call):

   ```
   python main.py --mode segments --dataset S1
   python main.py --mode ig2      --dataset S1
   python main.py --mode idg      --dataset S1
   python main.py --mode analysis --dataset S1 --attention both      # or ig2 / idg
                                   --dataset S1,S2,genai_tigs_biggan_7    # comma-separated, pools canonicalization -- analysis only
   python main.py --mode full-ig2 --dataset S1   # segments -> ig2 -> analysis(ig2), one call
   python main.py --mode full-idg --dataset S1   # segments -> idg -> analysis(idg), one call
   ```

   `--limit-images N` works with `segments`/`ig2`/`idg`/`full-*`. Since
   `segments`, `ig2`, and `idg` are mutually independent (per step 1-3 above),
   they can be run in any order or in parallel; `analysis` should be run
   last, once whichever of segments/attention it needs is done.

## Adding a new dataset

Add one file to `config/datasets/<dataset_id>.json` (see the existing files
for the two supported `lc_type`s) and, if the images aren't already under
`data/<dataset_id>/images/`, point `images_dir` at wherever they are. No
script changes needed for a dataset using an existing `lc_type`; a genuinely
new model architecture needs one new loader added to
`src/common/lc_models.py::load_wrapped_lc()`/`load_lc_for_ig2()`.

## Layout

```
src/            main.py (entry point) + pipeline scripts + src/common (shared logic) +
                src/third_party (vendored IG2/IDG algorithms, plus the B2T/EP-DNN-testing/
                Mimicry/GenAI-TIG baselines used by RQ1-RQ3)
config/         attention.json (shared hyperparameters) + datasets/*.json (per-dataset)
data/           per-dataset images + metadata.csv (image_id, image_path, output_value, label)
models/         model-under-test checkpoints referenced by datasets/*.json's lc_model_path
                (e.g. models/driving_model.ckpt, models/celeba_erm_seed<N>/best_model.pth)
script/         shell wrappers around main.py, one per mode
output/         <dataset>/<model>/<stage>/ -- stage is segments, ig2, idg, analysis_ig2, or analysis_idg;
                also output/model_repair/<DISC|B2T>/ for model-repair job checkpoints/logs
results/        reserved for RQ-level graphs only (cross-dataset/cross-run comparison figures) --
                separate from output/ above; see src/common/paths.py's RQ_RESULTS_DIR
```
