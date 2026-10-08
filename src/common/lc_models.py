"""Config-driven loading of the labeled classifier (LC) under test, for both
attention methods. Adding a new dataset means adding one config entry with an
EXISTING lc_type; a genuinely new model architecture needs one new loader
branch here (and a wrapper class if its output shape is new) -- that is the
one place model-specific code has to live, so ig2.py/idg.py/generate_analysis.py
never branch on dataset or model themselves.

Two lc_types: "sgsm_resnet34" (dict-output driving model, one fixed output
channel, postcondition-based pass/fail) and "imagenet_classifier" (plain
logit-tensor torchvision classifier, per-image target class from metadata,
pass/fail already given). Every wrapper's forward(x) returns (batch, 1)
regardless of the underlying LC's real output shape, so callers never need
to know which lc_type they got.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models as tv_models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224


# --- SGSM driving model -----------------------------------------------------
class ResnetModel(nn.Module):
    """SGSM's driving model: ResNet-34 backbone, two regression heads
    (acceleration, steering angle)."""

    def __init__(self):
        super().__init__()
        resnet34 = tv_models.resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
        self.resnet34 = nn.Sequential(*list(resnet34.children())[:-1])

        def head():
            return nn.Sequential(
                nn.Flatten(), nn.Dropout(0.5),
                nn.Linear(resnet34.fc.in_features, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 1), nn.Tanh(),
            )
        self.acceleration_head = head()
        self.steering_angle_head = head()

    def forward(self, x):
        features = self.resnet34(x)
        acceleration = torch.clip(self.acceleration_head(features), -1, 1)
        steering_angle = torch.clip(self.steering_angle_head(features), -1, 1)
        return {
            "acceleration": torch.clip(acceleration, 0, 1)[:, 0],
            "steering_angle": steering_angle[:, 0],
            "brake": torch.clip(acceleration, -1, 0)[:, 0],
            "acceleration_continuous": acceleration[:, 0],
        }


def load_sgsm_model(checkpoint_path: str, device) -> ResnetModel:
    model = ResnetModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model.eval().to(device)


# --- Output wrappers: normalize every LC to a (batch, 1) forward() ---------
class DictOutputWrapper(nn.Module):
    """sgsm_resnet34: model(x) returns a dict of named heads -- pick one."""

    def __init__(self, lc_model, output_key):
        super().__init__()
        self.lc_model, self.output_key = lc_model, output_key

    def forward(self, x):
        return self.lc_model(x)[self.output_key].unsqueeze(1)


class LogitOutputWrapper(nn.Module):
    """imagenet_classifier: model(x) returns (batch, n_classes) logits --
    preprocess INSIDE forward() so it stays differentiable end to end, then
    pick one class's logit."""

    def __init__(self, clf_model, target_class: int, device):
        super().__init__()
        self.clf_model, self.target_class = clf_model, target_class
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _resize_short_side(self, t):
        """Differentiable equivalent of transforms.Resize: scales the shorter
        edge to RESIZE_SHORT_SIDE, preserving aspect ratio (bilinear)."""
        _, _, h, w = t.shape
        if h <= w:
            new_h, new_w = RESIZE_SHORT_SIDE, max(RESIZE_SHORT_SIDE, round(w * RESIZE_SHORT_SIDE / h))
        else:
            new_w, new_h = RESIZE_SHORT_SIDE, max(RESIZE_SHORT_SIDE, round(h * RESIZE_SHORT_SIDE / w))
        return nn.functional.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def _center_crop(self, t):
        _, _, h, w = t.shape
        top, left = max(0, (h - CROP_SIZE) // 2), max(0, (w - CROP_SIZE) // 2)
        return t[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]

    def forward(self, x):
        x = self._center_crop(self._resize_short_side(x))
        x = (x / 255.0 - self.mean) / self.std
        return self.clf_model(x)[:, self.target_class].unsqueeze(1)


_IMAGENET_CLASSIFIERS = {
    "wide_resnet50_2": lambda: tv_models.wide_resnet50_2(weights=tv_models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
    "vgg19_bn": lambda: tv_models.vgg19_bn(weights=tv_models.VGG19_BN_Weights.IMAGENET1K_V1),
}


# --- Fine-tuned classifier checkpoints (e.g. DISC's Waterbirds ERM model) --
# For a model that was already fine-tuned elsewhere and saved as a whole
# torch.save(model, ...) object (not a state_dict) -- torch.load() returns a
# ready-to-use nn.Module directly, no architecture reconstruction needed.
# Preprocessing here matches DISC's own disc/dataset/transform.py::
# get_transform_cub() eval transform EXACTLY (square Resize((256,256)), not
# the aspect-preserving short-side resize _IMAGENET_CLASSIFIERS above use)
# -- required so PRISM attributes the same inputs the checkpoint was
# evaluated with. A genuinely different fine-tuned model's own preprocessing
# would need its own wrapper class, same as any new architecture.
#
# NOT every checkpoint is a whole-model object, though: B2T's own vendored
# gdro/erm_celeba.py (used for CelebA) saves {'model': model.state_dict()}
# instead (confirmed by inspecting the real checkpoint directly -- 320 keys,
# exact match to torchvision.models.resnet50's own state_dict key names, so
# no need to import B2T's own duplicate resnet.py implementation). Handle
# both shapes here rather than adding a CelebA-specific branch elsewhere.
def load_finetuned_model(checkpoint_path: str, device):
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()

    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    num_classes = state_dict["fc.weight"].shape[0]
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


class SquareResizeLogitWrapper(nn.Module):
    """Same role as LogitOutputWrapper (raw 0-255 CHW in, one class's logit
    out, (batch, 1)) but matching whichever eval transform the checkpoint
    was actually trained/evaluated under (config's "preprocess" key selects
    which -- see PREPROCESS_CONVENTIONS below):
    - "square_crop_256_224": DISC's CUB/Waterbirds transform
      (get_transform_cub) -- square-stretch Resize(256,256) -> CenterCrop(224).
    - "square_resize_224": DISC's ISIC transform (get_transform_ISIC) --
      square-stretch Resize(224,224) directly, no separate crop step. (That
      transform also applies ColorJitter(hue=0.2) even in its eval branch --
      a randomized op DISC's own code runs at eval time too. Deliberately
      NOT replicated here: a random transform has no single deterministic
      "correct" output to reproduce, and PRISM's own pass/fail check needs
      to be repeatable.)
    - "short_side_256_crop_224": B2T's gdro/data_loader.py CelebA transform
      (transform_valid in prepare_celeba) -- ASPECT-PRESERVING Resize(256)
      (shorter edge) -> CenterCrop(224), not a square stretch. Distinct
      resize_mode from the two above, since CelebA's own images (178x218)
      would visibly distort under a square-stretch resize.
    """

    def __init__(self, clf_model, target_class: int, device, resize_to: int, crop_size: int | None,
                 resize_mode: str = "stretch"):
        super().__init__()
        self.clf_model, self.target_class = clf_model, target_class
        self.resize_to, self.crop_size, self.resize_mode = resize_to, crop_size, resize_mode
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _resize(self, x):
        if self.resize_mode == "stretch":
            return nn.functional.interpolate(x, size=(self.resize_to, self.resize_to),
                                               mode="bilinear", align_corners=False)
        # "short_side": preserve aspect ratio, shorter edge -> resize_to (matches
        # torchvision.transforms.Resize(int) semantics).
        _, _, h, w = x.shape
        if h <= w:
            new_h, new_w = self.resize_to, max(self.resize_to, round(w * self.resize_to / h))
        else:
            new_w, new_h = self.resize_to, max(self.resize_to, round(h * self.resize_to / w))
        return nn.functional.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self._resize(x)
        if self.crop_size is not None:
            _, _, h, w = x.shape
            top, left = max(0, (h - self.crop_size) // 2), max(0, (w - self.crop_size) // 2)
            x = x[:, :, top:top + self.crop_size, left:left + self.crop_size]
        x = (x / 255.0 - self.mean) / self.std
        return self.clf_model(x)[:, self.target_class].unsqueeze(1)


# resize_to, crop_size (None = no separate crop step), resize_mode per named convention.
PREPROCESS_CONVENTIONS = {
    "square_crop_256_224": (256, CROP_SIZE, "stretch"),
    "square_resize_224": (224, None, "stretch"),
    "short_side_256_crop_224": (256, CROP_SIZE, "short_side"),
}


def get_target_class(cfg: dict, image_id, metadata_csv: Path) -> int:
    """"fixed:N" (a literal channel/class index) or "metadata:COLUMN" (looked
    up per image from that dataset's own metadata.csv)."""
    src = cfg["target_class_source"]
    if src.startswith("fixed:"):
        return int(src.split(":", 1)[1])
    if src.startswith("metadata:"):
        column = src.split(":", 1)[1]
        row = pd.read_csv(metadata_csv).set_index("image_id").loc[int(image_id)]
        return int(row[column])
    raise ValueError(f"Unrecognized target_class_source {src!r}")


def load_wrapped_lc(cfg: dict, image_id, metadata_csv: Path, device):
    """Returns (wrapped_model, target_class). wrapped_model.forward(x) always
    returns (batch, 1)."""
    target_class = get_target_class(cfg, image_id, metadata_csv)
    if cfg["lc_type"] == "sgsm_resnet34":
        lc_model = load_sgsm_model(cfg["lc_model_path"], device)
        wrapped = DictOutputWrapper(lc_model, cfg["output_key"])
    elif cfg["lc_type"] == "imagenet_classifier":
        if cfg["classifier_name"] not in _IMAGENET_CLASSIFIERS:
            raise ValueError(f"Unknown classifier_name {cfg['classifier_name']!r} -- "
                              f"add a loader to _IMAGENET_CLASSIFIERS")
        clf = _IMAGENET_CLASSIFIERS[cfg["classifier_name"]]().to(device).eval()
        wrapped = LogitOutputWrapper(clf, target_class, device)
    elif cfg["lc_type"] == "finetuned_classifier":
        clf = load_finetuned_model(cfg["lc_model_path"], device)
        resize_to, crop_size, resize_mode = PREPROCESS_CONVENTIONS[cfg.get("preprocess", "square_crop_256_224")]
        wrapped = SquareResizeLogitWrapper(clf, target_class, device, resize_to, crop_size, resize_mode)
    else:
        raise ValueError(f"Unknown lc_type {cfg['lc_type']!r} -- add a loader branch "
                          f"in load_wrapped_lc(), the one place a new architecture needs code")
    return wrapped.to(device).eval(), target_class


def load_input_tensor(cfg: dict, image_path, device) -> torch.Tensor:
    """sgsm_resnet34 needs an already-normalized tensor; imagenet_classifier
    needs a raw 0-255 CHW tensor since its wrapper normalizes internally
    (keeps the whole chain differentiable for IDG's backprop).

    For non-sgsm datasets, square-stretch resizes to cfg["raw_image_size"]
    first when set -- same purpose/convention as image_to_chw_float's own
    `size` param (ig2.py): IDG's own algorithm runs `steps` batched
    interpolation steps directly on this raw tensor BEFORE the LC wrapper's
    own internal resize, so for a dataset with large/variable native
    resolutions (e.g. ISIC, up to ~6700x4400) leaving it unset means IDG
    allocates its interpolation batch at that full native size -- easily
    tens of GB per image, real CUDA OOM territory. Datasets whose images are
    already small/uniform (SGSM, Waterbirds' own already-cropped photos)
    can leave raw_image_size unset, unchanged from before."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    if cfg["lc_type"] == "sgsm_resnet34":
        import torchvision.transforms as T
        tensor = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(img)
        return tensor.unsqueeze(0).to(device)
    if cfg.get("raw_image_size"):
        h, w = cfg["raw_image_size"]
        img = img.resize((w, h), Image.BILINEAR)
    arr = np.array(img).astype(np.float32).transpose(2, 0, 1)
    return torch.tensor(arr, device=device).unsqueeze(0)


def load_lc_for_ig2(cfg: dict, device):
    """IG2's gradient-descent path search perturbs RAW 0-255 pixels directly
    (clip_min_max=[0,255]), unlike IDG which works on the LC's normally-
    normalized input space -- so this returns the model UNwrapped, plus a
    normalize_fn(raw_0_255_tensor) applied inside IG2's own call_model_
    function, not before. Called ONCE per dataset (the model doesn't depend
    on which image is being attributed). Returns (model, normalize_fn,
    rep_module, extract_scalar_fn_factory). rep_module is the bottleneck
    submodule IG2 hooks for REP_LAYER_VALUES/REP_DISTANCE_GRADIENTS
    (config["rep_layer"], a dotted name under the model); calling
    extract_scalar_fn_factory(target_class) returns the function that pulls
    that one scalar output IG2 attributes against from the model's raw
    (dict or logit-tensor) output -- the only genuinely per-image part."""
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    if cfg["lc_type"] == "sgsm_resnet34":
        model = load_sgsm_model(cfg["lc_model_path"], device)

        def normalize_fn(x):
            return (x / 255.0 - mean_t) / std_t

        def extract_scalar_fn_factory(target_class):
            def extract_scalar_fn(out):
                return out[cfg["output_key"]]
            return extract_scalar_fn
    elif cfg["lc_type"] == "imagenet_classifier":
        if cfg["classifier_name"] not in _IMAGENET_CLASSIFIERS:
            raise ValueError(f"Unknown classifier_name {cfg['classifier_name']!r}")
        model = _IMAGENET_CLASSIFIERS[cfg["classifier_name"]]().to(device).eval()

        def _resize_short_side(t):
            _, _, h, w = t.shape
            if h <= w:
                new_h, new_w = RESIZE_SHORT_SIDE, max(RESIZE_SHORT_SIDE, round(w * RESIZE_SHORT_SIDE / h))
            else:
                new_w, new_h = RESIZE_SHORT_SIDE, max(RESIZE_SHORT_SIDE, round(h * RESIZE_SHORT_SIDE / w))
            return nn.functional.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)

        def _center_crop(t):
            _, _, h, w = t.shape
            top, left = max(0, (h - CROP_SIZE) // 2), max(0, (w - CROP_SIZE) // 2)
            return t[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]

        def normalize_fn(x):
            return (_center_crop(_resize_short_side(x)) / 255.0 - mean_t) / std_t

        def extract_scalar_fn_factory(target_class):
            def extract_scalar_fn(out):
                return out[:, target_class]
            return extract_scalar_fn
    elif cfg["lc_type"] == "finetuned_classifier":
        model = load_finetuned_model(cfg["lc_model_path"], device)
        resize_to, crop_size, resize_mode = PREPROCESS_CONVENTIONS[cfg.get("preprocess", "square_crop_256_224")]

        def _resize(x):
            if resize_mode == "stretch":
                return nn.functional.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
            _, _, h, w = x.shape
            if h <= w:
                new_h, new_w = resize_to, max(resize_to, round(w * resize_to / h))
            else:
                new_w, new_h = resize_to, max(resize_to, round(h * resize_to / w))
            return nn.functional.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        def normalize_fn(x):
            x = _resize(x)
            if crop_size is not None:
                _, _, h, w = x.shape
                top, left = max(0, (h - crop_size) // 2), max(0, (w - crop_size) // 2)
                x = x[:, :, top:top + crop_size, left:left + crop_size]
            return (x / 255.0 - mean_t) / std_t

        def extract_scalar_fn_factory(target_class):
            def extract_scalar_fn(out):
                return out[:, target_class]
            return extract_scalar_fn
    else:
        raise ValueError(f"Unknown lc_type {cfg['lc_type']!r}")

    model = model.to(device).eval()
    rep_module = dict(model.named_modules())[cfg["rep_layer"]]
    return model, normalize_fn, rep_module, extract_scalar_fn_factory


def image_to_chw_float(image_path, size: tuple | None = None) -> np.ndarray:
    """Raw (C, H, W) float32 in [0, 255], for IG2's own pixel-space search.

    size=(H, W), optional: square-stretch resize before returning (PIL
    bilinear). Needed for datasets whose source images aren't all the same
    native resolution -- IG2 stacks several of these into one array (the
    reference batch) up front, which requires a uniform shape. Datasets
    with fixed-size source images (SGSM, ImageNet crops) don't need this
    and should leave it None so existing behavior/output is unchanged."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.transpose(np.array(img), (2, 0, 1)).astype(np.float32)


# --- Pass/fail labeling (SGSM only -- imagenet datasets bring their own) ----
def _check_postcondition(value: float, postcondition: dict) -> bool:
    op, threshold = postcondition["operator"], postcondition["threshold"]
    return {"<=": value <= threshold, ">=": value >= threshold, "<": value < threshold,
            ">": value > threshold, "==": value == threshold}[op]


def ensure_metadata_csv(cfg: dict, images_dir: Path, metadata_csv: Path, device) -> pd.DataFrame:
    """Idempotent: if metadata_csv already exists, just load it. Otherwise
    (sgsm_resnet34 only) runs every image through the LC once, applies the
    dataset's postcondition, and writes image_id/image_path/output_value/label."""
    if metadata_csv.exists():
        return pd.read_csv(metadata_csv)
    if cfg["lc_type"] != "sgsm_resnet34":
        raise RuntimeError(f"{metadata_csv} does not exist and lc_type={cfg['lc_type']!r} datasets "
                            f"bring their own pre-generated metadata.csv -- nothing to compute here")

    image_paths = sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem))
    lc_model = load_sgsm_model(cfg["lc_model_path"], device)
    output_key = cfg["output_key"]
    rows = []
    with torch.no_grad():
        for i, path in enumerate(image_paths):
            tensor = load_input_tensor(cfg, path, device)
            value = lc_model(tensor)[output_key].item()
            rows.append({"image_id": i, "image_path": str(path), "output_key": output_key,
                         "output_value": value,
                         "label": "pass" if _check_postcondition(value, cfg["postcondition"]) else "fail"})
    df = pd.DataFrame(rows)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_csv, index=False)
    return df
