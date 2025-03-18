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
from utils._example_utils import (
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

def predictions(torch_func, kokkos_func, img, labels):
    pred1 = top3_possibilities(torch_func(img), labels)
    print("PyTorch prediction")
    print(pred1)
    pred2 = top3_possibilities(torch.from_numpy(kokkos_func(img.numpy())), labels)
    print("LAPIS prediction")
    print(pred2)
    # Return success if top class is correct, and its probability is close to torch's
    return pred1[0][0] == pred2[0][0] and math.fabs(pred1[0][1] - pred2[0][1]) < 0.001

img = load_and_preprocess_image_file("images/dog.jpg")
labels = load_labels()

resnet18 = models.resnet18(pretrained=True).eval()

imgPH = TensorPlaceholder([1, 3, 224, 224], torch.float32)
kModule = torchscript.compile(resnet18, imgPH, output_type="linalg-on-tensors")
kBackend = KokkosBackend.KokkosBackend()
kCompiledModule = kBackend.compile(kModule)

if predictions(resnet18.forward, kCompiledModule.forward, img, labels):
    print("Success, most-likely class and probability matched")
    sys.exit(0)
else:
    print("** Failed, most-likely class and probability did not match between torch and LAPIS")
    sys.exit(1)

