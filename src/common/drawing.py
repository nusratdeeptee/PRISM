"""Shared rendering: DINO boxes, SAM2 masks, and the per-image contact sheet."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [(255, 0, 0), (0, 200, 0), (0, 100, 255), (255, 150, 0), (200, 0, 200),
           (0, 200, 200), (255, 220, 0), (150, 75, 0)]

_FONT = None


def get_font():
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            _FONT = ImageFont.load_default()
    return _FONT


def draw_boxes(pil_image: Image.Image, objects: list) -> Image.Image:
    boxed = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(boxed)
    font = get_font()
    for i, obj in enumerate(objects):
        color = PALETTE[i % len(PALETTE)]
        x0, y0, x1, y1 = obj["box"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 2, max(0, y0 - 14)), f"{obj['label']} {obj['score']:.2f}", fill=color, font=font)
    return boxed


def draw_masks(pil_image: Image.Image, objects: list) -> Image.Image:
    base = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    font = get_font()
    for i, obj in enumerate(objects):
        mask = obj.get("mask")
        if mask is None:
            continue
        color = PALETTE[i % len(PALETTE)]
        mask_img = Image.fromarray((mask * 110).astype(np.uint8), mode="L")
        colored = Image.new("RGBA", base.size, color + (0,))
        colored.putalpha(mask_img)
        overlay = Image.alpha_composite(overlay, colored)
        ys, xs = np.where(mask)
        if len(xs):
            x0, y0 = int(xs.min()), int(ys.min())
            ImageDraw.Draw(overlay).text((x0 + 2, max(0, y0 - 14)), f"{obj['label']} {obj['score']:.2f}",
                                          fill=color + (255,), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


def build_contact_sheet(pil_image, highlighted_img, objects, rows_by_label, tags: list,
                         attention_name: str, pass_fail_label: str, alpha: float, out_path):
    """5 panels: original (+decision) / attention highlight / DINO boxes /
    SAM2 masks / objects that survived the permutation-significance filter."""
    thumb_h = 300
    decision = str(pass_fail_label).upper()

    def resize(im):
        w, h = im.size
        return im.resize((max(1, int(w * thumb_h / h)), thumb_h))

    chosen_objects = [o for o, r in zip(objects, rows_by_label) if r["is_significant"]]
    boxed = draw_boxes(pil_image, objects)
    masked = draw_masks(pil_image, objects)
    chosen_masked = draw_masks(pil_image, chosen_objects) if chosen_objects else pil_image.convert("RGB").copy()
    font = get_font()

    row_imgs = [resize(pil_image.convert("RGB")), resize(highlighted_img.convert("RGB")),
                resize(boxed), resize(masked), resize(chosen_masked)]
    labels = [f"ORIGINAL -- DECISION: {decision}", f"{attention_name.upper()} HIGHLIGHT",
              "GROUNDING DINO BOXES", "SAM2 MASKS", f"CHOSEN OBJECTS ({len(chosen_objects)}/{len(objects)})"]
    gap = 10
    row_w = sum(im.width for im in row_imgs) + gap * (len(row_imgs) + 1)
    label_h = 20

    chosen_labels = [o["label"] for o in chosen_objects]
    lines = [
        f"DECISION: {decision}",
        f"Qwen objects ({len(tags)}): {', '.join(tags)}",
        f"CHOSEN (p<{alpha}, {len(chosen_labels)}/{len(objects)}): "
        + (", ".join(chosen_labels) if chosen_labels else "(none significant)"),
        "",
    ]
    for r in sorted(rows_by_label, key=lambda r: r["permutation_pvalue"]):
        ms = f"{r['mass_share']:.3f}" if r["mass_share"] is not None else "N/A"
        ash = f"{r['area_share']:.3f}"
        peak_r = f"{r['peak_ratio']:.2f}" if r["peak_ratio"] is not None else "N/A"
        score = f"{r['score']:.2f}" if r["score"] is not None else "--"
        sig = "*" if r["is_significant"] else " "
        lines.append(f"{sig} {r['label']:<20} score={score:<5} area_share={ash:<6} "
                      f"mass_share={ms:<6} peak_ratio={peak_r:<6} p={r['permutation_pvalue']:.3f}")
    text_block = "\n".join(lines)
    text_h = 16 * (len(lines) + 1) + 20

    sheet = Image.new("RGB", (row_w, thumb_h + label_h + text_h), (255, 255, 255))
    d2 = ImageDraw.Draw(sheet)
    x = gap
    for im, lab in zip(row_imgs, labels):
        sheet.paste(im, (x, label_h))
        d2.text((x, 2), lab, fill=(0, 0, 0), font=font)
        x += im.width + gap
    d2.multiline_text((gap, thumb_h + label_h + 5), text_block, fill=(0, 0, 0), font=font, spacing=4)
    sheet.save(out_path)
