# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys
from pathlib import Path
import math
import torch
import torchvision.models as models
from torchvision import transforms
from torch_mlir import torchscript
from torch_mlir.compiler_utils import TensorPlaceholder
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from lapis import KokkosBackend
from PIL import Image

sys.path.append(str(Path(__file__).absolute().parent))
from _example_utils import (
    top3_possibilities,
    load_labels
)

def load_and_preprocess_image_file(path: str):
    img = Image.open(path).convert("RGB")
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

def load_multiple_images(paths):
    tensors = [load_and_preprocess_image_file(p) for p in paths]
    return torch.cat(tensors, dim=0)

def predictions(torch_func, kokkos_func, images, labels):
    torch_pred = torch_func(images)
    kokkos_pred = torch.from_numpy(kokkos_func(images.numpy()))
    success = True
    for i in range(3):
        print("Image", i, "top 3 predictions:")
        pred1 = top3_possibilities(torch_pred[i:i+1, :], labels)
        print("PyTorch prediction")
        print(pred1)
        pred2 = top3_possibilities(kokkos_pred[i:i+1, :], labels)
        print("LAPIS prediction")
        print(pred2)
        matched = pred1[0][0] == pred2[0][0] and math.fabs(pred1[0][1] - pred2[0][1]) < 0.001
        if not matched:
            success = False
    return success

images = load_multiple_images(['images/' + name + '.jpg' for name in ['goldfish', 'dog', 'cat']])
print("Loaded 3 images: overall tensor shape is", images.shape)
labels = load_labels()

resnet18 = models.resnet18(pretrained=True).eval()

# Batch dimension == -1 means dynamic.
imgPH = TensorPlaceholder([-1, 3, 224, 224], torch.float32)
kModule = torchscript.compile(resnet18, imgPH, output_type="linalg-on-tensors")
kBackend = KokkosBackend.KokkosBackend()
kCompiledModule = kBackend.compile(kModule)

if predictions(resnet18.forward, kCompiledModule.forward, images, labels):
    print("Success, most-likely classes and probabilities matched")
    sys.exit(0)
else:
    print("** Failed, most-likely classes and/or probabilities did not match between torch and LAPIS")
    sys.exit(1)

