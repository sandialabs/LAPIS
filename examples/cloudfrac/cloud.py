#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch_mlir
from torch_mlir.compiler_utils import TensorPlaceholder
from torch_mlir import torchscript
from lapis import KokkosBackend

class CldFracNet(nn.Module):
  def __init__(self, input_size, output_size, neuron_count=64):
    super(CldFracNet, self).__init__()
    # emulate cld_ice = (qi > 1e-5)
    self.ice1 = nn.Linear(input_size, neuron_count)
    self.ice2 = nn.Linear(neuron_count, output_size)
    # emulate cld_tot = max(cld_ice, cld_liq)
    self.tot1 = nn.Linear(input_size*2, neuron_count)
    self.tot2 = nn.Linear(neuron_count, output_size)
    # a relu for fun
    self.relu = nn.ReLU()
    # sigmoid for categorical ice output
    self.sigmoid = nn.Sigmoid()
    # Random weights
    nn.init.normal_(self.ice1.weight, mean=0, std=1)
    nn.init.normal_(self.ice2.weight, mean=0, std=1)
    nn.init.normal_(self.tot1.weight, mean=0, std=1)
    nn.init.normal_(self.tot2.weight, mean=0, std=1)

  def forward(self, qi, liq):
    # First, compute cld_ice from qi
    y11 = self.ice1(qi)
    y12 = self.relu(y11)
    y13 = self.ice2(y12)
    # Apply sigmoid to get probabilities
    y13_probabilities = self.sigmoid(y13)

    # During inference, use hard binary values
    y13_categorical = (y13_probabilities > 0.5).float()

    # Now compute cld_tot from cld_ice and cld_liq
    y21 = self.tot1(torch.cat((liq, y13_categorical), dim=0))
    y22 = self.relu(y21)
    y23 = self.tot2(y22)
    return y13_categorical, y23

def writeTensor(name, t):
    tn = t.detach().numpy()
    f = open(name, 'w')
    (m, n) = tn.shape
    for i in range(m):
        for j in range(n):
            f.write(str(tn[i,j]))
            f.write('\n')
    f.close()

def main ():
    # For this test, hard code nlevs, as well as pth file name/path
    nlevs = 72

    model = CldFracNet(nlevs,nlevs)

    dtype=torch.float32

    col = torch.ones((1,nlevs))

    qi = col
    liq = col

    # Sample random inputs, compute outputs, write both to file
    batch = 16
    in1 = torch.randn((batch, nlevs))
    in2 = torch.randn((batch, nlevs))
    out1 = torch.zeros((batch, nlevs))
    out2 = torch.zeros((batch, nlevs))

    for i in range(batch):
        (out1[i, :], out2[i, :]) = model.forward(in1[i, :], in2[i, :])

    ph = TensorPlaceholder([nlevs], torch.float32)
    #ph = TensorPlaceholder([batch, nlevs], torch.float32)

    writeTensor('../data/cloudfrac_in1.txt', in1)
    writeTensor('../data/cloudfrac_in2.txt', in2)
    writeTensor('../data/cloudfrac_out1.txt', out1)
    writeTensor('../data/cloudfrac_out2.txt', out2)

    mlir_module = torchscript.compile(model, (ph, ph), output_type='linalg-on-tensors')
    with open("cloudfrac.mlir",'w') as fd:
        fd.write(str(mlir_module))

if __name__ == "__main__":
    main()
