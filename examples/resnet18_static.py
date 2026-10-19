# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys
from pathlib import Path
import math
import torch
from torchvision import transforms
from torch_mlir import fx
from lapis import KokkosBackend
from PIL import Image

sys.path.append(str(Path(__file__).absolute().parent))
from utils._example_utils import (
    top3_possibilities,
    load_labels,
    load_resnet18_weights,
)
# Pretrained weights and labels are loaded via load_resnet18_weights() and
# load_labels(). To use real weights offline, download:
#   https://download.pytorch.org/models/resnet18-f37072fd.pth
#   https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt
# and place them in examples/data/ as resnet18-f37072fd.pth and imagenet-classes.txt.

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
print(img.shape)

# Uncomment the following lines to write the raw values of preprocessed input
# tensor to a file, for use in standalone C++ driver
#print("Writing img (preprocessed) values to dog_preprocessed.txt")
#with open('dog_preprocessed.txt', 'w') as f:
#    for i in range(3):
#        for j in range(224):
#            for k in range(224):
#                f.write(str(float(img[0, i, j, k])))
#                f.write('\n')
#print("Done")

labels = load_labels()

resnet18 = load_resnet18_weights().eval()
kModule = fx.export_and_import(resnet18, img, output_type="linalg-on-tensors", func_name='forward')
kBackend = KokkosBackend.KokkosBackend()
kCompiledModule = kBackend.compile(kModule)

if predictions(resnet18.forward, kCompiledModule.forward, img, labels):
    print("Success, most-likely class and probability matched")
    sys.exit(0)
else:
    print("** Failed, most-likely class and probability did not match between torch and LAPIS")
    sys.exit(1)

