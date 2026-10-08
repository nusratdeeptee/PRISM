"""
PRISM's own parameterized BigGAN (genai_tigs) ImageNet generation driver.

Adapted from GRIT/baseline/genai_tigs's
experiments/imagenet_experiments/imagenet_gan/conditional_gan/tig_biggan_imagenet.py
(a SINVAD-style latent-space genetic search against a frozen BigGAN-deep-256
generator) -- an unmodified copy of that file is kept alongside this one as
tig_biggan_imagenet_vendor_reference.py for diffing. The algorithm itself
(fitness function, crossover/mutation loop, GA hyperparameters) is
reproduced verbatim below; everything new here is: wrapping it in a
generate() function, CLI args (class/count/output-dir/file-prefix) in place
of the vendor's hardcoded globals + sed-patching convention, and using
wide_resnet50_2 as the classifier by default -- matching the classifier
swap already done in genai_tigs/slurm/run_imagenet_biggan_10class.sh so
BigGAN and mimicry probe the exact same system-under-test (both driver's
own comment there explains why: apples-to-apples cross-technique
comparison over the same 10 ImageNet classes).

Natively saves real PNG files during generation (no post-hoc extraction
needed, unlike mimicry) -- originals under original_images/or_images/,
perturbed/final candidates under perturb_images/.

Usage:
    <genai_tigs-venv>/bin/python generate.py \\
        --class-idx 0 --num-images 480 --output-dir /path/to/biggan_ai_tigs_0 \\
        --file-prefix task03_
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from tqdm import trange


def save_image(image_tensor: torch.Tensor, filename: str) -> None:
    """Converts a tensor to an image and saves it (verbatim from vendor)."""
    img = image_tensor.clone().detach().to("cpu").float().numpy()
    if img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.transpose(img, (1, 2, 0))
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    Image.fromarray(img).save(filename)


def _calculate_fitness(logit: np.ndarray, label: int) -> float:
    """Verbatim from vendor tig_biggan_imagenet.py."""
    expected_logit = logit[label]
    best_index1, best_index2 = np.argsort(-logit)[:2]
    best_but_not_expected = best_index2 if best_index1 == label else best_index1
    return expected_logit - logit[best_but_not_expected]


def generate(
    class_idx: int,
    num_images: int,
    output_dir: str,
    file_prefix: str = "",
    classifier_name: str = "wide_resnet50_2",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_images_dir = os.path.join(output_dir, "original_images")
    img_pil_dir = os.path.join(original_images_dir, "or_images")
    prtb_img_dir = os.path.join(output_dir, "perturb_images")
    os.makedirs(img_pil_dir, exist_ok=True)
    os.makedirs(prtb_img_dir, exist_ok=True)

    preprocess_for_classifier = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def preprocess_for_saving(image_tensor: torch.Tensor) -> Image.Image:
        image = (image_tensor + 1) / 2
        image = image.squeeze(0)
        return transforms.ToPILImage()(image.cpu())

    def preprocess_image(image_tensor: torch.Tensor) -> torch.Tensor:
        image = preprocess_for_saving(image_tensor)
        image = preprocess_for_classifier(image)
        return image.unsqueeze(0).to(device)

    G = BigGAN.from_pretrained("biggan-deep-256").to(device)
    classifier = torch.hub.load("pytorch/vision:v0.10.0", classifier_name, pretrained=True).to(device)
    classifier.eval()
    G.eval()

    num_classes = 1000
    gen_num = 250
    pop_size = 25
    best_left = 10
    min_val = -1.99990940093994
    max_val = 1.99996650218964
    initial_perturbation_size = 0.000799975180625916

    truncation = 1
    expected_label = class_idx
    saved_images = 0

    for _ in trange(num_images):
        seed = torch.randint(0, 10000, (1,)).item()
        torch.manual_seed(seed)
        noise_vector = truncated_noise_sample(batch_size=1, dim_z=128, truncation=truncation, seed=seed)
        original_latent = torch.from_numpy(noise_vector).to(device)
        original_latent = torch.clamp(original_latent, min=min_val, max=max_val)

        original_label_onehot = torch.zeros(1, num_classes).to(device)
        original_label_onehot[:, expected_label] = 1
        with torch.no_grad():
            original_image = G(original_latent, original_label_onehot, truncation).to(device)
            original_image = preprocess_image(original_image)
        original_logit = classifier(original_image).squeeze().detach().cpu().numpy()
        original_label = np.argmax(original_logit).item()

        if original_label != expected_label:
            print(f"Label mismatch for image {saved_images}: expected {expected_label}, got {original_label}", flush=True)
            continue

        tag = f"{file_prefix}{saved_images}"
        original_image_path = os.path.join(img_pil_dir, f"{tag}_X{original_label}.png")
        save_image(original_image.squeeze(0), original_image_path)
        saved_images += 1
        print(f"Success: image {saved_images} classified correctly as expected class {original_label}.", flush=True)

        init_pop = [
            original_latent.unsqueeze(0) + initial_perturbation_size * torch.randn_like(original_latent)
            for _ in range(pop_size)
        ]
        pop_labels = torch.tensor([original_label] * pop_size)
        pop_class_vectors = F.one_hot(pop_labels, num_classes).float().to(device)

        best_fitness_score = np.inf
        best_image_tensor = None
        now_pop = init_pop
        prev_best = np.inf
        perturbation_size = initial_perturbation_size
        g_idx = 0

        for g_idx in range(gen_num):
            indivs_lv = torch.cat(now_pop, dim=0).to(device).view(pop_size, -1)
            indivs_labels = pop_class_vectors.view(pop_size, -1).to(device)
            with torch.no_grad():
                gen_imgs = G(indivs_lv, indivs_labels, truncation=truncation).to(device)
                gen_imgs = torch.cat([preprocess_image(img) for img in gen_imgs])

            all_logits = classifier(gen_imgs).squeeze().detach().cpu().numpy()
            fitness_scores = [_calculate_fitness(all_logits[k], original_label) for k in range(pop_size)]

            selected_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[-best_left:]
            current_min_index = int(np.argmin(fitness_scores))
            current_min_fitness = fitness_scores[current_min_index]
            if current_min_fitness < best_fitness_score:
                best_fitness_score = current_min_fitness
                best_image_tensor = gen_imgs[current_min_index]

            now_best = float(np.min(fitness_scores))
            parent_pop = [now_pop[i] for i in selected_indices]

            if now_best < 0:
                break
            elif now_best == prev_best:
                perturbation_size *= 2
            else:
                perturbation_size = initial_perturbation_size

            k_pop = []
            for _ in range(pop_size - best_left):
                mom_idx, pop_idx = np.random.choice(best_left, size=2, replace=False)
                spl_idx = np.random.choice(128, size=1)[0]
                k_gene = torch.cat([parent_pop[mom_idx][:, :spl_idx], parent_pop[pop_idx][:, spl_idx:]], dim=1)
                diffs = (k_gene != original_latent).float()
                k_gene += perturbation_size * torch.randn_like(k_gene) * diffs
                k_pop.append(k_gene)

            now_pop = parent_pop + k_pop
            prev_best = now_best
            now_pop = [torch.clamp(t, min=min_val, max=max_val) for t in now_pop]

        if best_image_tensor is None:
            print("image is none", flush=True)
            continue

        with torch.no_grad():
            final_logits = classifier(best_image_tensor.unsqueeze(0)).squeeze().detach().cpu().numpy()
        predicted_best_label = int(np.argmax(final_logits))

        best_image_pil_path = os.path.join(
            prtb_img_dir, f"{tag}_iteration_{g_idx + 1}_X{original_label}_Y{predicted_best_label}.png"
        )
        save_image(best_image_tensor.squeeze(0), best_image_pil_path)

    print(f"[generate] class={class_idx}: saved {saved_images} original+perturbed PNG pairs to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM BigGAN (genai_tigs) ImageNet generation driver.")
    parser.add_argument("--class-idx", type=int, required=True, help="ImageNet class index (0-9).")
    parser.add_argument("--num-images", type=int, required=True, help="Number of seed attempts to draw for this run (accept rate is below 100%%, see vendor's own RQ1 note).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write images into.")
    parser.add_argument("--file-prefix", type=str, default="", help="Prefix for output filenames (keeps parallel array tasks collision-free).")
    parser.add_argument("--classifier", type=str, default="wide_resnet50_2", help="torchvision classifier name (default matches mimicry's SUT for cross-technique comparison).")
    args = parser.parse_args()

    generate(args.class_idx, args.num_images, args.output_dir, args.file_prefix, args.classifier)
