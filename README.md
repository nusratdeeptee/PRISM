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

## Data Availability

Due to the extensive size of the full experimental data, we are only able to
make a subset publicly available: https://zenodo.org/records/23248457

### Waterbirds

| Seeds | DISC own concepts |
|---|---|
| 0, 1, 2 | bamboo, banded, beach, blackness, blotchy, blueness, blurriness, braided, bridge, bubbly, bumpy, bush, canopy, chequered, cobwebbed, concrete, cracked, crosshatched, crystalline, dotted, earth, fibrous, field, flecked, flower, flowerpot, fluorescent, forest, freckled, frilly, gauzy, granite, grass, greenness, grid, grooved, ground, harbor, hill, honeycombed, interlaced, knitted, lacelike, lake, laminate, leather, lined, marbled, matted, meshed, metal, mountain, muzzle, paisley, palm, path, perforated, pitted, plant, pleated, polka_dots, porous, potholed, redness, river, sand, scaly, sea, smeared, snow, spiralled, sprinkled, stained, stratified, stripes, studded, swirly, tree, veined, waffled, water, whiteness, woven, wrinkled, zigzagged |

| Seed | DISC+PRISM | DISC+PRISM manually excluded | B2T own-discovered -- Land | B2T own-discovered -- Water | B2T+PRISM -- Land | B2T+PRISM -- Water |
|---|---|---|---|---|---|---|
| 0 | forest, leaves, branch, bamboo, flower, tree, grass, fish, water, mountain, seal, ship, house, rock, sky, person, boat, wave, lake, ocean, cloud | Unknown Background, bird, duck, hand, people, seagull | bamboo forest, bamboo, forest, woods, trees, garden, tree | seagull, beach, boat, water, lake, sunset, rocks | bamboo, forest, tree, branch | ocean, seal, wave, seagull, boat, ship, water, fish, duck |
| 1 | sign, branch, bamboo, hand, building, leaves, fish, wing, mountain, rock, tree, sky, grass, flower, water, house, wave, ocean, seal, boat, ship, cloud, surfer | Unknown Background, bird, boat ship, duck, people, person, sailboat, seagull | bamboo forest, bamboo, forest, woods, trees, garden, tree | seagull, beach, boat, water, lake, fish | bamboo, tree, branch | ocean, sailboat, seal, seagull, surfer, wave, boat, boat ship, ship, water, fish |
| 2 | leaves, bamboo, house, tree, building, grass, flower, person, rock, sky, female_face, water, ocean, ship, seal, hand, hat, pier, boat, wave, cloud | Unknown Background, bird, duck, head, people, seagull, woman | bamboo forest, bamboo, forest, woods, rainforest, trees, garden, tree | seagull, beach, water, sunset, lake, rocks | bamboo, tree | ocean, seagull, seal, wave, pier, boat, ship, water, clouds |

### CelebA

| Seed | B2T own-discovered | B2T+PRISM |
|---|---|---|
| 0 | *(none)* | man, man green shirt, man person |
| 1 | man | man |
| 2 | *(none)* | man, man person |

### MetaShift

| Seeds | DISC own concepts |
|---|---|
| 0, 1, 2 | air_conditioner, airplane, apron, arm, armchair, ashcan, awning, back, back_pillow, bag, balcony, bamboo, banded, bannister, base, basket, bathroom_s, bathtub, beach, beak, bed, bedclothes, bedroom_s, bench, bicycle, bird, blackness, blind, blotchy, blueness, blurriness, board, body, book, bookcase, bottle, bowl, box, braided, brick, bridge, bubbly, bucket, building, bumper, bumpy, bus, bush, cabinet, can, candlestick, canopy, cap, car, cardboard, carpet, cat, ceiling, ceramic, chain_wheel, chair, chandelier, chequered, chest_of_drawers, chimney, clock, coach, cobwebbed, coffee_table, column, computer, concrete, counter, countertop, cow, cracked, crosshatched, crystalline, cup, curtain, cushion, desk, dining_room_s, dog, door, door_frame, dotted, double_door, drawer, drinking_glass, ear, earth, engine, exhaust_hood, eye, eyebrow, fabric, fan, faucet, female_face, fence, fibrous, field, figurine, fireplace, flag, flecked, floor, flower, flowerpot, fluorescent, food, foot, footboard, forest, frame, freckled, frilly, gauzy, glass, granite, grass, greenness, grid, grooved, ground, hair, hand, handle, handle_bar, harbor, head, headboard, headlight, hill, honeycombed, horse, house, inside_arm, interlaced, jar, keyboard, knitted, knob, lacelike, lake, laminate, lamp, leather, leg, lid, light, lined, loudspeaker, male_face, manhole, marbled, matted, meshed, metal, microwave, minibike, mirror, motorbike, mountain, mouse, mouth, muzzle, napkin, neck, nose, ottoman, outside_arm, oven, pack, painted, painting, paisley, palm, pane, paper, path, paw, pedestal, perforated, person, pillar, pillow, pipe, pitted, plant, plate, pleated, polka_dots, porous, potholed, redness, refrigerator, river, sand, scaly, sea, smeared, snow, sofa, spiralled, sprinkled, stained, stairs, stratified, street_s, stripes, studded, swirly, toilet, tree, veined, waffled, water, whiteness, woven, wrinkled, zigzagged |

| Seed | DISC+PRISM | DISC+PRISM manually excluded |
|---|---|---|
| 0 | bottle, laptop, person, wall, window, chair, mirror, bowl, keyboard, female_face, pillow, couch, toilet, remote_control, bookshelf, table, coffee_table, picture_frame, frisbee, curtain, desk, book, candle, wine_glass, vase | shelf |
| 1 | laptop, bottle, person, table, shelf, wall, chair, bowl, pillow, couch, toilet, keyboard, remote_control, drawer, lamp, window, candle, mirror, female_face, bookshelf, frisbee, curtain, tile, coffee_table, book | desk |
| 2 | laptop, table, coffee_table, chair, bottle, person, pillow, couch, male_face, toilet, keyboard, window, bowl, remote_control, mouse, picture_frame, wall, shelf, female_face, candle, desk, lamp, book, vase, curtain | *(none)* |

### ISIC

| Trap-set | DISC own concepts | DISC+PRISM | DISC+PRISM manually excluded |
|---|---|---|---|
| 1 | blurriness, braided, bubbly, chequered, concrete, crosshatched, flecked, granite, lacelike, laminate, matted, meshed, pleated, redness, scaly, studded, veined | bubbles, hair, ruler, water_droplet | skin, Unknown Background, water droplets, mole, stain, blood, bruise |
| 2 | concrete, cracked, crystalline, fibrous, flecked, frilly, gauzy, granite, grid, marbled, paisley, pleated, potholed, spiralled, stained, stripes, swirly | bubbles, hair, line, ruler, skin, water_droplet | water droplets, Unknown Background, mole, background, bruise |
| 3 | banded, braided, concrete, gauzy, grid, honeycombed, lacelike, marbled, matted, metal, paisley, pleated, porous, sprinkled, veined, waffled | bubbles, dirt, hair, line, petri_dish, ruler, skin, surface, water_droplet | Unknown Background, mole, water droplets, stain, blood spot, spot, skin mole |
| 4 | blueness, braided, concrete, crosshatched, dotted, flecked, gauzy, granite, lacelike, laminate, marbled, redness, smeared, stained, swirly, whiteness | bubbles, hair, ruler, skin, water_droplet | Unknown Background, water droplets, bruise, stain, mole, skin mole, blood stain, blood spot, blood |
| 5 | blurriness, cobwebbed, concrete, crosshatched, flecked, frilly, gauzy, lacelike, laminate, leather, paisley, smeared, stained, stripes, veined | bubbles, hair, plastic_container, ruler, skin, water_droplet | surface, blood spot, water droplets, stain, tumor, Unknown Background, mole, spot |
