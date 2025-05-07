import mala
import torch
from torch_mlir import torchscript
from torch_mlir.compiler_utils import TensorPlaceholder
from lapis import KokkosBackend
import shutil

parameters, network, data_handler, predictor = mala.Predictor.load_run("be_model")
network.eval()
print("Loaded model from be_model.zip:")
print(network)
descrPH = TensorPlaceholder([8748, 91], torch.float32)
module = torchscript.compile(network, descrPH, output_type="linalg-on-tensors")
backend = KokkosBackend.KokkosBackend()
backend.compile(module)
print("Copying generated C++ file to forward_snap.hpp...")
shutil.copyfile('lapis_package/lapis_package_module.cpp', 'forward_snap.hpp')
