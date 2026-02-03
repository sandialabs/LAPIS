#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch_mlir
from torch_mlir.compiler_utils import TensorPlaceholder
from torch_mlir import fx
from torch_mlir import torchscript
from lapis import KokkosBackend

class SimpleNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.layer1 = nn.Linear(input_size, hidden_size)
    self.layer2 = nn.Linear(hidden_size, output_size)
    self.activation = nn.Sigmoid()
    nn.init.normal_(self.layer1.weight, mean=0.0, std=1.0, generator=None)
    nn.init.normal_(self.layer1.bias, mean=0.0, std=1.0, generator=None)
    nn.init.normal_(self.layer2.weight, mean=0.0, std=1.0, generator=None)
    nn.init.normal_(self.layer2.bias, mean=0.0, std=1.0, generator=None)

  def forward(self, x):
    x = self.layer1(x)
    x = self.activation(x)
    x = self.layer2(x)
    pred = self.activation(x)
    return pred

class SimpleNN_Loss(nn.Module):
  def __init__(self):
    super().__init__()
    self.activation = nn.Sigmoid()
    self.loss = nn.MSELoss()

  def forward(self, weight1, bias1, weight2, bias2, x, target):
    # forward for this module just computes the loss (it doesn't return predictions)
    y = (x @ weight1.transpose(0,1)) + bias1
    y = self.activation(y)
    z = (y @ weight2.transpose(0,1)) + bias2
    z = self.activation(z)
    return self.loss(z, target)

def main ():
    input_size = 10
    hidden_size = 20
    output_size = 5
    # loss gradient will simulate a training step with this batch size
    batch = 8

    # torch automatically initializes parameters randomly
    model = SimpleNN(input_size, hidden_size, output_size)
    modelLoss = SimpleNN_Loss().eval()
    #with torch.no_grad():
    #    modelLoss.layer1.weight.copy_(model.layer1.weight)
    #    modelLoss.layer1.bias.copy_(model.layer1.bias)
    #    modelLoss.layer2.weight.copy_(model.layer2.weight)
    #    modelLoss.layer2.bias.copy_(model.layer2.bias)

    #print("Shape of layer1 weights:", modelLoss.layer1.weight.shape)
    #print("Shape of layer1 bias   :", modelLoss.layer1.bias.shape)
    #print("Shape of layer2 weights:", modelLoss.layer2.weight.shape)
    #print("Shape of layer2 bias   :", modelLoss.layer2.bias.shape)

    dummyInput = torch.ones((batch, input_size), requires_grad=False)
    dummyTarget = torch.ones((batch, output_size), requires_grad=False)

    #lossExported = fx.export_and_import(
    #    modelLoss,
    #    dummyInput, dummyTarget,
    #    output_type="linalg-on-tensors",
    #    func_name='forward',
    #)
    #lossKokkos = backend.compile(lossExported)

    Loss = nn.MSELoss()
    out = model.forward(dummyInput)
    loss = Loss(out, dummyTarget)
    loss.backward()

    #print("Loss                :", loss.item())
    #print("Loss from LAPIS     :", lossKokkos.forward(dummyInput, dummyTarget))

    print("Grad of layer 1 weights:", model.layer1.weight.grad)
    print("Grad of layer 1 biases:", model.layer1.bias.grad)
    print("Grad of layer 2 weights:", model.layer2.weight.grad)
    print("Grad of layer 2 biases:", model.layer2.bias.grad)

    # Can't convert parameter tensors (with gradients) to NumPy,
    # so make clean clones of them to pass to LAPIS module
    weight1 = model.layer1.weight.clone().detach()
    bias1 = model.layer1.bias.clone().detach()
    weight2 = model.layer2.weight.clone().detach()
    bias2 = model.layer2.bias.clone().detach()

    # Compile the loss function using LAPIS, with reverse-mode gradient of parameters wrt loss
    lossModule = fx.export_and_import(
        modelLoss,
        weight1, bias1, weight2, bias2, dummyInput, dummyTarget,
        output_type="linalg-on-tensors",
        func_name='loss',
    )
    backend = KokkosBackend.KokkosBackend(dump_mlir=True)
    lossKokkos = backend.compile(lossModule) # 'loss', 'grad_loss', ['active'], ['active', 'active', 'active', 'active', 'const', 'const'])
    print("Loss from SimpleNN_Loss.forward:", modelLoss.forward(weight1, bias1, weight2, bias2, dummyInput, dummyTarget).item())
    print("Loss from LAPIS                :", lossKokkos.loss(weight1, bias1, weight2, bias2, dummyInput, dummyTarget))
    #gradModule = backend.reverse_diff_compile(lossModule, 'loss', 'grad_loss', ['active'], ['active', 'active', 'active', 'active', 'const', 'const'])
    #modelKokkos = backend.compile_rev(lossExported)

    #inputPH = TensorPlaceholder([batch, input_size], torch.float32)
    #lossModule = torchscript.compile(Loss, (inputPH, inputPH), output_type="linalg-on-tensors")
    #exported = torchscript.compile(model, (inputPH, inputPH), output_type="linalg-on-tensors")

    # x is a random model input
    #x = torch.randn((batch, input_size), requires_grad=False)
    #print("Input:")
    #print(x)
    #out = model(x)
    #print("Output:")
    #print(out)
    #target = torch.zeros((batch, output_size))
    #target[0, 3] = 1
    ##Loss = nn.MSELoss()
    #Loss = nn.L1Loss()
    #loss = Loss(out, target)
    #print("Loss:")
    #print(loss)
    #loss.backward()

    #backend = KokkosBackend.KokkosBackend(decompose_tensors=True, dump_mlir=True)
    #module_kokkos = backend.compile(, 'myfunc', 'grad_myfunc', ['dupnoneed'], ['dup', 'dup', 'dup'])

    #kBackend = KokkosBackend.KokkosBackend()
    #kCompiledModule = kBackend.compile(kModule)

if __name__ == "__main__":
    main()

