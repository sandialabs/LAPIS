# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from PIL import Image
from pathlib import Path
import requests

import torch
from torchvision import transforms


DEFAULT_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
)
DEFAULT_LABEL_URL = (
    "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt"
)

# Directory for cached data files (weights, labels, etc.)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_LABEL_FILE = DATA_DIR / "imagenet-classes.txt"
LOCAL_WEIGHTS_FILE = DATA_DIR / "resnet18-f37072fd.pth"


def load_and_preprocess_image(url: str = DEFAULT_IMAGE_URL):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }
    img = Image.open(requests.get(url, headers=headers, stream=True).raw).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


def load_labels(url: str = DEFAULT_LABEL_URL):
    if LOCAL_LABEL_FILE.exists():
        with open(LOCAL_LABEL_FILE, "r") as f:
            classes_text = f.read()
    else:
        try:
            classes_text = requests.get(
                url=url,
                stream=True,
            ).text
        except Exception:
            classes_text = ""
    labels = [line.strip() for line in classes_text.splitlines()] if classes_text else [f"class_{i}" for i in range(1000)]
    return labels


def load_resnet18_weights():
    """Load ResNet18 pretrained weights from local file if available.

    Looks for examples/data/resnet18-f37072fd.pth first. If that file
    does not exist, falls back to torchvision's default download behavior.
    If no network is available, returns a model with random weights
    (using a fixed seed for reproducibility).
    """
    import torchvision.models as models

    if LOCAL_WEIGHTS_FILE.exists():
        state_dict = torch.load(LOCAL_WEIGHTS_FILE, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model = models.resnet18(weights=None)
        model.load_state_dict(state_dict)
        return model
    else:
        try:
            return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            torch.manual_seed(42)
            return models.resnet18(weights=None)


def top3_possibilities(res, labels):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3
