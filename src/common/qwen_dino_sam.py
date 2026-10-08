"""Object discovery: Qwen2.5-VL names every visible object, Grounding DINO
grounds those names into boxes, SAM2 turns each box into a mask. Qwen is
asked to be exhaustive and explicitly tolerate hallucination -- a
hallucinated phrase with no real visual referent simply fails to produce a
box, so it costs a wasted DINO forward pass, not a false detection."""
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

QWEN_PROMPT = (
    "You are an expert object detector and this is an object detection "
    "task. Carefully examine this image and identify every visible object "
    "in it. You must identify small, partial, background, and minor "
    "objects along with main objects. Be exhaustive: it is better to "
    "include something you are only fairly confident about than to leave "
    "it out. For each object, give only its name as a short noun "
    "phrase. Do not describe, explain, or group objects together -- "
    "output ONLY a single comma-separated list of object names, with no "
    "numbering, no extra sentences, and no repetition."
)


def load_qwen(model_name: str):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    return model, AutoProcessor.from_pretrained(model_name), device


def generate_qwen_objects(model, processor, device, pil_image: Image.Image, max_tags: int) -> tuple:
    """One Qwen call -> parsed, de-duplicated, capped list of object-name
    phrases, plus the raw response text."""
    import torch
    from qwen_vl_utils import process_vision_info

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        pil_image.save(tmp.name)
        tmp.close()
        messages = [{"role": "user", "content": [{"type": "image", "image": tmp.name},
                                                    {"type": "text", "text": QWEN_PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    raw = raw.strip()
    if ":" in raw[:80]:  # drop a leading "Here is the list of objects:" preamble, if any
        raw = raw.split(":", 1)[1]
    phrases, seen = [], set()
    for p in re.split(r",|\n|;", raw):
        p = p.strip().strip(".").lower()
        p = re.sub(r"^\d+[\).\s]*", "", p)
        if p and p not in seen and len(p) < 40:
            seen.add(p)
            phrases.append(p)
    return phrases[:max_tags], raw


def load_grounding_dino(model_name: str, device: str):
    import torch
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    model = GroundingDinoForObjectDetection.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    return model, AutoProcessor.from_pretrained(model_name)


def detect_objects(model, processor, device, pil_image: Image.Image, phrases: list,
                    box_threshold: float, text_threshold: float) -> list:
    import torch
    text_prompt = ". ".join(phrases) + "."
    # Real bug (found 2026-09-21, 8/19867 CelebA images): with no truncation,
    # a long/verbose phrase list (max_tags allows up to 40) can tokenize past
    # the model's own config.max_text_len (256) -- the model's internal
    # truncation logic (modeling_grounding_dino.py's encoder path, ~line 2089)
    # truncates several tensors to max_text_len but NOT the one used in the
    # earlier text-feature-enhancer masked_fill (~line 1323), so the two end
    # up different lengths and it crashes with a tensor-size mismatch instead
    # of silently truncating. Truncating up front, before the model ever sees
    # more than max_text_len tokens, avoids the inconsistent internal path
    # entirely rather than patching around it.
    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt",
                        truncation=True, max_length=model.config.max_text_len).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    w, h = pil_image.size
    results = processor.post_process_grounded_object_detection(
        outputs, input_ids=inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[(h, w)])[0]
    labels = results.get("text_labels", results.get("labels"))
    return [{"label": str(label), "score": float(score), "box": [float(x) for x in box.tolist()]}
            for box, score, label in zip(results["boxes"], results["scores"], labels)]


def load_sam2(model_name: str, device: str):
    from transformers import Sam2Model, Sam2Processor
    return Sam2Model.from_pretrained(model_name).to(device), Sam2Processor.from_pretrained(model_name)


def masks_for_boxes(sam_model, sam_processor, device, pil_image: Image.Image, boxes: list) -> list:
    """Box-prompted SAM2, one mask per box, best-of-3 by predicted IoU."""
    import torch
    if not boxes:
        return []
    image_inputs = sam_processor(images=pil_image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)
    original_sizes = image_inputs["original_sizes"]
    with torch.no_grad():
        embeds = sam_model.get_image_embeddings(pixel_values)
    del pixel_values

    n = len(boxes)
    box_inputs = sam_processor(images=[pil_image] * n, input_boxes=[[b] for b in boxes], return_tensors="pt")
    batch_emb = [e.expand(n, *([-1] * (e.dim() - 1))) for e in embeds]
    with torch.no_grad():
        outputs = sam_model(image_embeddings=batch_emb, input_boxes=box_inputs["input_boxes"].to(device),
                             multimask_output=True)
    masks_list = sam_processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes.expand(n, -1))
    scores = outputs.iou_scores.cpu().numpy()

    masks = []
    for i in range(n):
        best = int(scores[i, 0].argmax())
        masks.append(masks_list[i][0, best].numpy().astype(bool))
    return masks


def count_ungrounded(tags: list, detections: list) -> int:
    """A tag counts as grounded if it matches a detection label exactly or
    as a whole word inside a merged one (DINO can combine multiple tags into
    one label, e.g. "rooster dog") -- or vice versa."""
    det_labels = [d["label"].lower() for d in detections]
    return sum(1 for t in (t.lower() for t in tags)
               if not any(t == l or t in l.split() or l in t.split() for l in det_labels))
