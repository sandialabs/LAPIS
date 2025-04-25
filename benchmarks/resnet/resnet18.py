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
import shutil

resnet18 = models.resnet18(pretrained=True).eval()

# Batch dimension == -1 means dynamic.
imgPH = TensorPlaceholder([-1, 3, 224, 224], torch.float32)
kModule = torchscript.compile(resnet18, imgPH, output_type="linalg-on-tensors")
kBackend = KokkosBackend.KokkosBackend()

print("Lowering model, generating C++ and compiling C++...")
kCompiledModule = kBackend.compile(kModule)
print("Moving generated C++ to resnet.hpp...")
shutil.move('lapis_package/lapis_package_module.cpp', 'resnet.hpp')

