import argparse
import importlib
import os
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import clip

from sklearn.metrics import classification_report
from tqdm import tqdm

from data.celeba import CelebA
from data.waterbirds import Waterbirds

import celeba_templates
import waterbirds_templates

# GRIT/PRISM additions (2026-09-16):
# 1. Explicit download_root -- upstream's bare clip.load('RN50', 'cuda', jit=False) defaults to
#    ~/.cache/clip (home dir). This project has already hit home-quota exhaustion from exactly
#    this class of unset-cache-dir mistake (see feedback_slurm... memory), so route it to a
#    configurable cache dir instead, same convention already used for HF_HOME elsewhere in this
#    project. Override with the PRISM_CLIP_CACHE env var (e.g. to point at a shared/quota-safe
#    location again); defaults to a repo-relative directory so this stays self-contained.
_PRISM_ROOT = Path(__file__).resolve().parents[4]
CLIP_DOWNLOAD_ROOT = os.environ.get("PRISM_CLIP_CACHE", str(_PRISM_ROOT / ".cache" / "clip"))


def main(args):
    model, preprocess = clip.load('RN50', 'cuda', jit=False, download_root=CLIP_DOWNLOAD_ROOT)

    crop = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
    transform = transforms.Compose([crop, preprocess])

    if args.dataset == 'waterbirds':
        data_dir = os.path.join(args.data_dir, 'waterbird_complete95_forest2water2')
        train_dataset = Waterbirds(data_dir=data_dir, split='train', transform=transform)
        # GRIT/PRISM addition: --concepts_module lets the caller swap in an alternate
        # class_keywords_all (e.g. waterbirds_templates_prism) without touching this file
        # again or duplicating the ~80 CLIP prompt templates / class_templates, which are
        # imported unchanged from waterbirds_templates by every alternate module.
        concepts_mod = importlib.import_module(args.concepts_module)
        templates = concepts_mod.templates
        class_templates = concepts_mod.class_templates
        class_keywords_all = concepts_mod.class_keywords_all
    elif args.dataset == 'celeba':
        data_dir = os.path.join(args.data_dir, 'celeba')
        train_dataset = CelebA(data_dir=data_dir, split='train', transform=transform)
        # GRIT/PRISM addition: --concepts_module (default 'waterbirds_templates', i.e. not
        # passed) is honored for celeba too, so an alternate CelebA module such as
        # celeba_templates_prism can be swapped in. Unset -> upstream's celeba_templates.
        if args.concepts_module != 'waterbirds_templates':
            celeba_mod = importlib.import_module(args.concepts_module)
        else:
            celeba_mod = celeba_templates
        templates = celeba_mod.templates
        class_templates = celeba_mod.class_templates
        class_keywords_all = celeba_mod.class_keywords_all
    else:
        raise NotImplementedError

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=4, drop_last=False)
    temperature = 0.02  # redundant parameter

    with torch.no_grad():
        zeroshot_weights = []
        for class_keywords in class_keywords_all:
            texts = [template.format(class_template.format(class_keyword)) for template in templates for class_template in class_templates for class_keyword in class_keywords]
            texts = clip.tokenize(texts).cuda()

            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()

    preds_minor, preds, targets_minor = [], [], []
    with torch.no_grad():
        for (image, (target, target_g, target_s), _) in tqdm(train_dataloader):
            image = image.cuda()
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ zeroshot_weights / temperature

            probs = logits.softmax(dim=-1).cpu()
            conf, pred = torch.max(probs, dim=1)

            if args.dataset == 'waterbirds':
                # minor group if
                # (target, target_s) == (0, 1): landbird on water background
                # (target, target_s) == (1, 0): waterbird on land background
                is_minor_pred = (((target == 0) & (pred == 1)) | ((target == 1) & (pred == 0))).long()
                is_minor = (((target == 0) & (target_s == 1)) | ((target == 1) & (target_s == 0))).long()
            if args.dataset == 'celeba':
                # minor group if
                # (target, target_s) == (1, 1): blond man
                is_minor_pred = ((target == 1) & (pred == 1)).long()
                is_minor = ((target == 1) & (target_s == 1)).long()

            preds_minor.append(is_minor_pred)
            preds.append(pred)
            targets_minor.append(is_minor)

    preds_minor, preds, targets_minor = torch.cat(preds_minor), torch.cat(preds), torch.cat(targets_minor)

    print(classification_report(targets_minor, preds_minor))

    # Save pseudo labels
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(preds, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='celeba', choices=['celeba', 'waterbirds'])
    parser.add_argument('--data_dir', default='/data')
    parser.add_argument('--save_path', default='./pseudo_bias/celeba.pt')
    parser.add_argument('--concepts_module', default='waterbirds_templates',
                        help="Module (in this directory) providing templates/class_templates/"
                             "class_keywords_all for the waterbirds branch. Default matches "
                             "upstream's own hardcoded waterbirds_templates.")

    args = parser.parse_args()
    main(args)
