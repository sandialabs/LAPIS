import argparse
import os
from pathlib import Path
import shutil
import subprocess


MODULE = r"""
module {
  func.func @abx(%a: tensor<2x3xf32>, %b: tensor<3x5xf32>,
                 %x: tensor<5xf32>) -> tensor<2xf32> {
    %zero = arith.constant 0.0 : f32
    %matrix_empty = tensor.empty() : tensor<2x5xf32>
    %matrix_zero = linalg.fill ins(%zero : f32)
        outs(%matrix_empty : tensor<2x5xf32>) -> tensor<2x5xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<3x5xf32>)
      outs(%matrix_zero : tensor<2x5xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2x5xf32>

    %vector_empty = tensor.empty() : tensor<2xf32>
    %vector_zero = linalg.fill ins(%zero : f32)
        outs(%vector_empty : tensor<2xf32>) -> tensor<2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%ab, %x : tensor<2x5xf32>, tensor<5xf32>)
      outs(%vector_zero : tensor<2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>
    return %result : tensor<2xf32>
  }

  func.func @profitable_multi_output(
      %a: tensor<2x3xf32>, %b: tensor<3x5xf32>, %x: tensor<5xf32>,
      %z: tensor<2xf32>) -> (tensor<2xf32>, tensor<f32>) {
    %zero = arith.constant 0.0 : f32
    %matrix_empty = tensor.empty() : tensor<2x5xf32>
    %matrix_zero = linalg.fill ins(%zero : f32)
        outs(%matrix_empty : tensor<2x5xf32>) -> tensor<2x5xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x3xf32>, tensor<3x5xf32>)
      outs(%matrix_zero : tensor<2x5xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2x5xf32>

    %vector_empty = tensor.empty() : tensor<2xf32>
    %vector_zero = linalg.fill ins(%zero : f32)
        outs(%vector_empty : tensor<2xf32>) -> tensor<2xf32>
    %abx = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%ab, %x : tensor<2x5xf32>, tensor<5xf32>)
      outs(%vector_zero : tensor<2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2xf32>

    %scalar_empty = tensor.empty() : tensor<f32>
    %scalar_zero = linalg.fill ins(%zero : f32)
        outs(%scalar_empty : tensor<f32>) -> tensor<f32>
    %dot = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%abx, %z : tensor<2xf32>, tensor<2xf32>)
      outs(%scalar_zero : tensor<f32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<f32>
    return %abx, %dot : tensor<2xf32>, tensor<f32>
  }

  func.func @batched_abx(
      %a: tensor<2x2x3xf32>, %b: tensor<2x3x5xf32>,
      %x: tensor<2x5xf32>) -> tensor<2x2xf32> {
    %zero = arith.constant 0.0 : f32
    %matrix_empty = tensor.empty() : tensor<2x2x5xf32>
    %matrix_zero = linalg.fill ins(%zero : f32)
        outs(%matrix_empty : tensor<2x2x5xf32>) -> tensor<2x2x5xf32>
    %ab = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%a, %b : tensor<2x2x3xf32>, tensor<2x3x5xf32>)
      outs(%matrix_zero : tensor<2x2x5xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2x2x5xf32>

    %result_empty = tensor.empty() : tensor<2x2xf32>
    %result_zero = linalg.fill ins(%zero : f32)
        outs(%result_empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%ab, %x : tensor<2x2x5xf32>, tensor<2x5xf32>)
      outs(%result_zero : tensor<2x2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<2x2xf32>
    return %result : tensor<2x2xf32>
  }

  // One dense PCG iteration. The two producer-consumer regions mirror the
  // paper's Ap/p^T Ap and Jacobi-preconditioner/r^T z fusion candidates. The
  // vector updates intentionally remain ordinary linalg.generic operations so
  // this exercises candidate boundaries as well as successful fusion.
  func.func @pcg_step(
      %a: tensor<4x4xf64>, %p: tensor<4xf64>, %r: tensor<4xf64>,
      %z: tensor<4xf64>, %x: tensor<4xf64>, %dinv: tensor<4xf64>)
      -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>,
          tensor<4xf64>, tensor<f64>, tensor<f64>) {
    %vector_zero = arith.constant dense<0.0> : tensor<4xf64>
    %scalar_zero = arith.constant dense<0.0> : tensor<f64>

    %rz = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%r, %z : tensor<4xf64>, tensor<4xf64>)
      outs(%scalar_zero : tensor<f64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<f64>

    %ap = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %p : tensor<4x4xf64>, tensor<4xf64>)
      outs(%vector_zero : tensor<4xf64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<4xf64>

    %pap = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%p, %ap : tensor<4xf64>, tensor<4xf64>)
      outs(%scalar_zero : tensor<f64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<f64>

    %rz_value = tensor.extract %rz[] : tensor<f64>
    %pap_value = tensor.extract %pap[] : tensor<f64>
    %alpha = arith.divf %rz_value, %pap_value : f64

    %xnext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%x, %p : tensor<4xf64>, tensor<4xf64>)
      outs(%vector_zero : tensor<4xf64>) {
    ^bb0(%x_value: f64, %p_value: f64, %unused: f64):
      %scaled = arith.mulf %alpha, %p_value : f64
      %updated = arith.addf %x_value, %scaled : f64
      linalg.yield %updated : f64
    } -> tensor<4xf64>

    %rnext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%r, %ap : tensor<4xf64>, tensor<4xf64>)
      outs(%vector_zero : tensor<4xf64>) {
    ^bb0(%r_value: f64, %ap_value: f64, %unused: f64):
      %scaled = arith.mulf %alpha, %ap_value : f64
      %updated = arith.subf %r_value, %scaled : f64
      linalg.yield %updated : f64
    } -> tensor<4xf64>

    %znext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%rnext, %dinv : tensor<4xf64>, tensor<4xf64>)
      outs(%vector_zero : tensor<4xf64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<4xf64>

    %rznext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%rnext, %znext : tensor<4xf64>, tensor<4xf64>)
      outs(%scalar_zero : tensor<f64>) {
    ^bb0(%lhs: f64, %rhs: f64, %acc: f64):
      %product = arith.mulf %lhs, %rhs : f64
      %sum = arith.addf %acc, %product : f64
      linalg.yield %sum : f64
    } -> tensor<f64>

    %rznext_value = tensor.extract %rznext[] : tensor<f64>
    %beta = arith.divf %rznext_value, %rz_value : f64
    %pnext = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%znext, %p : tensor<4xf64>, tensor<4xf64>)
      outs(%vector_zero : tensor<4xf64>) {
    ^bb0(%z_value: f64, %p_value: f64, %unused: f64):
      %scaled = arith.mulf %beta, %p_value : f64
      %updated = arith.addf %z_value, %scaled : f64
      linalg.yield %updated : f64
    } -> tensor<4xf64>

    return %xnext, %rnext, %pnext, %znext, %ap, %pap, %rznext
        : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>,
          tensor<4xf64>, tensor<f64>, tensor<f64>
  }

  // Unnormalized linear-attention core. The source association materializes
  // QK^T with shape 4x6. Exact contraction planning instead forms K^T V with
  // shape 3x2, halving multiplication work from 120 to 60.
  func.func @linear_attention_core(
      %q: tensor<4x3xf32>, %k: tensor<6x3xf32>, %v: tensor<6x2xf32>)
      -> tensor<4x2xf32> {
    %score_zero = arith.constant dense<0.0> : tensor<4x6xf32>
    %scores = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%q, %k : tensor<4x3xf32>, tensor<6x3xf32>)
      outs(%score_zero : tensor<4x6xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<4x6xf32>

    %output_zero = arith.constant dense<0.0> : tensor<4x2xf32>
    %output = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%scores, %v : tensor<4x6xf32>, tensor<6x2xf32>)
      outs(%output_zero : tensor<4x2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<4x2xf32>
    return %output : tensor<4x2xf32>
  }

  // The vector intermediate is internalized, its common loop is fused with
  // the reduction, and scalar forwarding removes the buffer.
  func.func @fused_dot(
      %a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>)
      -> tensor<f32> {
    %zero0 = arith.constant dense<0.0> : tensor<4xf32>
    %lhs = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%zero0 : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<4xf32>

    %scalar_zero = arith.constant dense<0.0> : tensor<f32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%lhs, %c : tensor<4xf32>, tensor<4xf32>)
      outs(%scalar_zero : tensor<f32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %product = arith.mulf %x, %y : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<f32>
    return %result : tensor<f32>
  }
}
"""


DRIVER_SOURCE = r"""
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

template <typename T, int Rank>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  std::int64_t offset;
  std::int64_t sizes[Rank];
  std::int64_t strides[Rank];
};

template <typename T>
struct StridedMemRefType<T, 0> {
  T *basePtr;
  T *data;
  std::int64_t offset;
};

extern "C" void lapis_initialize();
extern "C" void lapis_finalize();
extern "C" void py_abx(StridedMemRefType<float, 1> **result,
                         StridedMemRefType<float, 2> *a,
                         StridedMemRefType<float, 2> *b,
                         StridedMemRefType<float, 1> *x);
extern "C" void py_profitable_multi_output(
    StridedMemRefType<float, 1> **abx,
    StridedMemRefType<float, 0> **dot,
    StridedMemRefType<float, 2> *a,
    StridedMemRefType<float, 2> *b,
    StridedMemRefType<float, 1> *x,
    StridedMemRefType<float, 1> *z);
extern "C" void py_batched_abx(StridedMemRefType<float, 2> **result,
                                StridedMemRefType<float, 3> *a,
                                StridedMemRefType<float, 3> *b,
                                StridedMemRefType<float, 2> *x);
extern "C" void py_pcg_step(
    StridedMemRefType<double, 1> **xnext,
    StridedMemRefType<double, 1> **rnext,
    StridedMemRefType<double, 1> **pnext,
    StridedMemRefType<double, 1> **znext,
    StridedMemRefType<double, 1> **ap,
    StridedMemRefType<double, 0> **pap,
    StridedMemRefType<double, 0> **rznext,
    StridedMemRefType<double, 2> *a,
    StridedMemRefType<double, 1> *p,
    StridedMemRefType<double, 1> *r,
    StridedMemRefType<double, 1> *z,
    StridedMemRefType<double, 1> *x,
    StridedMemRefType<double, 1> *dinv);
extern "C" void py_linear_attention_core(
    StridedMemRefType<float, 2> **result,
    StridedMemRefType<float, 2> *q,
    StridedMemRefType<float, 2> *k,
    StridedMemRefType<float, 2> *v);
extern "C" void py_fused_dot(
    StridedMemRefType<float, 0> **result,
    StridedMemRefType<float, 1> *a,
    StridedMemRefType<float, 1> *b,
    StridedMemRefType<float, 1> *c);

template <std::size_t Rank, typename T>
StridedMemRefType<T, Rank>
makeDescriptor(T *data, const std::array<std::int64_t, Rank> &sizes) {
  StridedMemRefType<T, Rank> descriptor{};
  descriptor.basePtr = data;
  descriptor.data = data;
  descriptor.offset = 0;
  std::int64_t stride = 1;
  for (std::size_t reverse = Rank; reverse > 0; --reverse) {
    const std::size_t dimension = reverse - 1;
    descriptor.sizes[dimension] = sizes[dimension];
    descriptor.strides[dimension] = stride;
    stride *= sizes[dimension];
  }
  return descriptor;
}

struct InputCase {
  std::array<float, 6> a;
  std::array<float, 15> b;
  std::array<float, 5> x;
  std::array<float, 2> z;
};

std::array<InputCase, 3> makeInputCases() {
  std::array<InputCase, 3> cases{};
  cases[0] = {{{1.25F, -2.0F, 0.5F, 3.0F, 0.75F, -1.5F}},
              {{2.0F, -1.0F, 0.25F, 3.0F, -0.5F,
                0.5F, 4.0F, -2.0F, 1.5F, 2.25F,
                -3.0F, 0.75F, 1.25F, -1.0F, 5.0F}},
              {{1.5F, -2.0F, 0.75F, 3.0F, -1.25F}},
              {{2.0F, -0.5F}}};

  for (std::size_t i = 0; i < cases[1].a.size(); ++i)
    cases[1].a[i] = static_cast<float>((static_cast<int>(i) * 5 + 3) % 9 - 4) * 0.35F;
  for (std::size_t i = 0; i < cases[1].b.size(); ++i)
    cases[1].b[i] = static_cast<float>((static_cast<int>(i) * 7 + 2) % 13 - 6) * 0.2F;
  for (std::size_t i = 0; i < cases[1].x.size(); ++i)
    cases[1].x[i] = static_cast<float>((static_cast<int>(i) * 3 + 1) % 7 - 3) * 0.4F;
  cases[1].z = {{-1.25F, 2.5F}};

  cases[2] = {{{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F}},
              {{1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                6.0F, 7.0F, 8.0F, 9.0F, 10.0F,
                11.0F, 12.0F, 13.0F, 14.0F, 15.0F}},
              {{0.0F, 1.0F, 0.0F, -1.0F, 2.0F}},
              {{-3.0F, 4.0F}}};
  return cases;
}

std::array<float, 2> evaluateOriginal(const InputCase &input) {
  std::array<float, 10> ab{};
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      for (std::size_t k = 0; k < 3; ++k)
        ab[i * 5 + j] += input.a[i * 3 + k] * input.b[k * 5 + j];
    }
  }

  std::array<float, 2> abx{};
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 5; ++j)
      abx[i] += ab[i * 5 + j] * input.x[j];
  }
  return abx;
}

bool approximatelyEqual(float actual, float expected) {
  constexpr float absoluteTolerance = 2.0e-5F;
  constexpr float relativeTolerance = 2.0e-5F;
  return std::abs(actual - expected) <=
         absoluteTolerance + relativeTolerance * std::abs(expected);
}

float readVectorElement(const StridedMemRefType<float, 1> &descriptor,
                        std::size_t index) {
  return descriptor.data[descriptor.offset +
                         static_cast<std::int64_t>(index) *
                             descriptor.strides[0]];
}

bool checkBatchedAbx() {
  std::array<float, 12> a{};
  std::array<float, 30> b{};
  std::array<float, 10> x{};
  for (std::size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<float>((static_cast<int>(i) * 5 + 1) % 11 - 5) * 0.3F;
  for (std::size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<float>((static_cast<int>(i) * 7 + 3) % 17 - 8) * 0.2F;
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<float>((static_cast<int>(i) * 3 + 2) % 9 - 4) * 0.4F;

  std::array<float, 20> ab{};
  std::array<float, 4> expected{};
  for (std::size_t batch = 0; batch < 2; ++batch) {
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j < 5; ++j) {
        for (std::size_t k = 0; k < 3; ++k) {
          ab[(batch * 2 + i) * 5 + j] +=
              a[(batch * 2 + i) * 3 + k] *
              b[(batch * 3 + k) * 5 + j];
        }
        expected[batch * 2 + i] +=
            ab[(batch * 2 + i) * 5 + j] * x[batch * 5 + j];
      }
    }
  }

  auto aDescriptor = makeDescriptor<3>(a.data(), {{2, 2, 3}});
  auto bDescriptor = makeDescriptor<3>(b.data(), {{2, 3, 5}});
  auto xDescriptor = makeDescriptor<2>(x.data(), {{2, 5}});
  StridedMemRefType<float, 2> result{};
  auto *resultPointer = &result;
  py_batched_abx(&resultPointer, &aDescriptor, &bDescriptor, &xDescriptor);

  bool passed = true;
  for (std::size_t batch = 0; batch < 2; ++batch) {
    for (std::size_t i = 0; i < 2; ++i) {
      float actual = result.data[
          result.offset + static_cast<std::int64_t>(batch) * result.strides[0] +
          static_cast<std::int64_t>(i) * result.strides[1]];
      float reference = expected[batch * 2 + i];
      if (!approximatelyEqual(actual, reference)) {
        std::cerr << "batched ABx mismatch at (" << batch << ", " << i
                  << "): got " << actual << ", expected " << reference
                  << '\n';
        passed = false;
      }
    }
  }
  return passed;
}

bool checkLinearAttentionCore() {
  std::array<float, 12> q{};
  std::array<float, 18> k{};
  std::array<float, 12> v{};
  for (std::size_t i = 0; i < q.size(); ++i)
    q[i] = static_cast<float>((static_cast<int>(i) * 5 + 2) % 13 - 6) * 0.2F;
  for (std::size_t i = 0; i < k.size(); ++i)
    k[i] = static_cast<float>((static_cast<int>(i) * 7 + 1) % 17 - 8) * 0.15F;
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = static_cast<float>((static_cast<int>(i) * 3 + 4) % 11 - 5) * 0.25F;

  std::array<float, 24> scores{};
  std::array<float, 8> expected{};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      for (std::size_t d = 0; d < 3; ++d)
        scores[i * 6 + j] += q[i * 3 + d] * k[j * 3 + d];
      for (std::size_t e = 0; e < 2; ++e)
        expected[i * 2 + e] += scores[i * 6 + j] * v[j * 2 + e];
    }
  }

  auto qDescriptor = makeDescriptor<2>(q.data(), {{4, 3}});
  auto kDescriptor = makeDescriptor<2>(k.data(), {{6, 3}});
  auto vDescriptor = makeDescriptor<2>(v.data(), {{6, 2}});
  StridedMemRefType<float, 2> result{};
  auto *resultPointer = &result;
  py_linear_attention_core(&resultPointer, &qDescriptor, &kDescriptor,
                           &vDescriptor);

  bool passed = true;
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t e = 0; e < 2; ++e) {
      const float actual = result.data[
          result.offset + static_cast<std::int64_t>(i) * result.strides[0] +
          static_cast<std::int64_t>(e) * result.strides[1]];
      const float reference = expected[i * 2 + e];
      if (!approximatelyEqual(actual, reference)) {
        std::cerr << "linear-attention mismatch at (" << i << ", " << e
                  << "): got " << actual << ", expected " << reference
                  << '\n';
        passed = false;
      }
    }
  }
  return passed;
}

bool checkFusedDot() {
  std::array<float, 4> a = {{1.0F, -2.0F, 0.5F, 3.0F}};
  std::array<float, 4> b = {{0.25F, 1.5F, -1.0F, 2.0F}};
  std::array<float, 4> c = {{2.0F, -0.5F, 4.0F, 1.25F}};
  float expected = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i)
    expected += (a[i] + b[i]) * c[i];

  auto aDescriptor = makeDescriptor<1>(a.data(), {{4}});
  auto bDescriptor = makeDescriptor<1>(b.data(), {{4}});
  auto cDescriptor = makeDescriptor<1>(c.data(), {{4}});
  StridedMemRefType<float, 0> result{};
  auto *resultPointer = &result;
  py_fused_dot(&resultPointer, &aDescriptor, &bDescriptor, &cDescriptor);
  const float actual = result.data[result.offset];
  if (approximatelyEqual(actual, expected))
    return true;
  std::cerr << "fused-dot mismatch: got " << actual << ", expected "
            << expected << '\n';
  return false;
}

double readPcgVectorElement(const StridedMemRefType<double, 1> &descriptor,
                            std::size_t index) {
  return descriptor.data[descriptor.offset +
                         static_cast<std::int64_t>(index) *
                             descriptor.strides[0]];
}

bool approximatelyEqualDouble(double actual, double expected) {
  constexpr double absoluteTolerance = 1.0e-11;
  constexpr double relativeTolerance = 1.0e-11;
  return std::abs(actual - expected) <=
         absoluteTolerance + relativeTolerance * std::abs(expected);
}

bool checkPcgStep() {
  std::array<double, 16> a = {{
      4.0, 1.0, 0.0, 0.0,
      1.0, 3.0, 1.0, 0.0,
      0.0, 1.0, 2.0, 1.0,
      0.0, 0.0, 1.0, 2.0,
  }};
  std::array<double, 4> p = {{1.0, -0.5, 0.75, 1.25}};
  std::array<double, 4> r = {{2.0, -1.0, 0.5, 1.5}};
  std::array<double, 4> dinv = {{0.25, 1.0 / 3.0, 0.5, 0.5}};
  std::array<double, 4> z{};
  std::array<double, 4> x = {{0.1, -0.2, 0.3, -0.4}};
  for (std::size_t i = 0; i < z.size(); ++i)
    z[i] = dinv[i] * r[i];

  double rz = 0.0;
  std::array<double, 4> ap{};
  for (std::size_t i = 0; i < 4; ++i) {
    rz += r[i] * z[i];
    for (std::size_t j = 0; j < 4; ++j)
      ap[i] += a[i * 4 + j] * p[j];
  }
  double pap = 0.0;
  for (std::size_t i = 0; i < 4; ++i)
    pap += p[i] * ap[i];
  const double alpha = rz / pap;

  std::array<double, 4> expectedX{};
  std::array<double, 4> expectedR{};
  std::array<double, 4> expectedZ{};
  double expectedRz = 0.0;
  for (std::size_t i = 0; i < 4; ++i) {
    expectedX[i] = x[i] + alpha * p[i];
    expectedR[i] = r[i] - alpha * ap[i];
    expectedZ[i] = dinv[i] * expectedR[i];
    expectedRz += expectedR[i] * expectedZ[i];
  }
  const double beta = expectedRz / rz;
  std::array<double, 4> expectedP{};
  for (std::size_t i = 0; i < 4; ++i)
    expectedP[i] = expectedZ[i] + beta * p[i];

  auto aDescriptor = makeDescriptor<2>(a.data(), {{4, 4}});
  auto pDescriptor = makeDescriptor<1>(p.data(), {{4}});
  auto rDescriptor = makeDescriptor<1>(r.data(), {{4}});
  auto zDescriptor = makeDescriptor<1>(z.data(), {{4}});
  auto xDescriptor = makeDescriptor<1>(x.data(), {{4}});
  auto dinvDescriptor = makeDescriptor<1>(dinv.data(), {{4}});

  StridedMemRefType<double, 1> xnextResult{};
  StridedMemRefType<double, 1> rnextResult{};
  StridedMemRefType<double, 1> pnextResult{};
  StridedMemRefType<double, 1> znextResult{};
  StridedMemRefType<double, 1> apResult{};
  StridedMemRefType<double, 0> papResult{};
  StridedMemRefType<double, 0> rznextResult{};
  auto *xnextPointer = &xnextResult;
  auto *rnextPointer = &rnextResult;
  auto *pnextPointer = &pnextResult;
  auto *znextPointer = &znextResult;
  auto *apPointer = &apResult;
  auto *papPointer = &papResult;
  auto *rznextPointer = &rznextResult;
  py_pcg_step(&xnextPointer, &rnextPointer, &pnextPointer, &znextPointer,
              &apPointer, &papPointer, &rznextPointer, &aDescriptor,
              &pDescriptor, &rDescriptor, &zDescriptor, &xDescriptor,
              &dinvDescriptor);

  bool passed = true;
  auto checkVector = [&](const char *name,
                         const StridedMemRefType<double, 1> &actual,
                         const std::array<double, 4> &expected) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const double value = readPcgVectorElement(actual, i);
      if (!approximatelyEqualDouble(value, expected[i])) {
        std::cerr << "PCG " << name << " mismatch at " << i << ": got "
                  << value << ", expected " << expected[i] << '\n';
        passed = false;
      }
    }
  };
  checkVector("xnext", xnextResult, expectedX);
  checkVector("rnext", rnextResult, expectedR);
  checkVector("pnext", pnextResult, expectedP);
  checkVector("znext", znextResult, expectedZ);
  checkVector("Ap", apResult, ap);
  if (!approximatelyEqualDouble(papResult.data[papResult.offset], pap)) {
    std::cerr << "PCG p^T A p mismatch: got "
              << papResult.data[papResult.offset] << ", expected " << pap
              << '\n';
    passed = false;
  }
  if (!approximatelyEqualDouble(rznextResult.data[rznextResult.offset],
                                expectedRz)) {
    std::cerr << "PCG rnext^T znext mismatch: got "
              << rznextResult.data[rznextResult.offset] << ", expected "
              << expectedRz << '\n';
    passed = false;
  }
  return passed;
}

int main() {
  lapis_initialize();
  bool passed = true;

  const auto cases = makeInputCases();
  for (std::size_t caseNumber = 0; caseNumber < cases.size(); ++caseNumber) {
    InputCase input = cases[caseNumber];
    auto a = makeDescriptor<2>(input.a.data(), {{2, 3}});
    auto b = makeDescriptor<2>(input.b.data(), {{3, 5}});
    auto x = makeDescriptor<1>(input.x.data(), {{5}});
    auto z = makeDescriptor<1>(input.z.data(), {{2}});

    const auto expectedAbx = evaluateOriginal(input);
    const float expectedDot =
        expectedAbx[0] * input.z[0] + expectedAbx[1] * input.z[1];

    StridedMemRefType<float, 1> abxResult{};
    auto *abxResultPointer = &abxResult;
    py_abx(&abxResultPointer, &a, &b, &x);
    for (std::size_t i = 0; i < expectedAbx.size(); ++i) {
      const float actual = readVectorElement(abxResult, i);
      if (!approximatelyEqual(actual, expectedAbx[i])) {
        std::cerr << "ABx mismatch in case " << caseNumber << ", element "
                  << i << ": got " << actual << ", expected "
                  << expectedAbx[i] << '\n';
        passed = false;
      }
    }

    StridedMemRefType<float, 1> multiAbxResult{};
    StridedMemRefType<float, 0> dotResult{};
    auto *multiAbxResultPointer = &multiAbxResult;
    auto *dotResultPointer = &dotResult;
    py_profitable_multi_output(&multiAbxResultPointer, &dotResultPointer,
                               &a, &b, &x, &z);
    for (std::size_t i = 0; i < expectedAbx.size(); ++i) {
      const float actual = readVectorElement(multiAbxResult, i);
      if (!approximatelyEqual(actual, expectedAbx[i])) {
        std::cerr << "multi-output ABx mismatch in case " << caseNumber
                  << ", element " << i << ": got " << actual
                  << ", expected " << expectedAbx[i] << '\n';
        passed = false;
      }
    }
    const float actualDot = dotResult.data[dotResult.offset];
    if (!approximatelyEqual(actualDot, expectedDot)) {
      std::cerr << "dot mismatch in case " << caseNumber << ": got "
                << actualDot << ", expected " << expectedDot << '\n';
      passed = false;
    }
  }

  passed = checkBatchedAbx() && passed;
  passed = checkLinearAttentionCore() && passed;
  passed = checkFusedDot() && passed;
  passed = checkPcgStep() && passed;

  lapis_finalize();
  if (!passed)
    return 1;
  std::cout << "All algebraic kernel fusion numerical checks passed.\n";
  return 0;
}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lapis-opt", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def skip_if_runtime_is_unavailable():
    missing = []
    for variable in ("KOKKOS_ROOT", "SUPPORT_LIB"):
        if not os.environ.get(variable):
            missing.append(variable)
    for executable in ("lapis-translate", "cmake"):
        if shutil.which(executable) is None:
            missing.append(executable)
    if missing:
        print("SKIP: missing numerical-test prerequisites: " + ", ".join(missing))
        raise SystemExit(77)


def transform_module(lapis_opt):
    completed = subprocess.run(
        [lapis_opt, "--algebraic-kernel-fusion"],
        input=MODULE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("algebraic transformation failed:\n" + completed.stderr)
    # The logical K^T V result is 3x2. Stable index ordering materializes its
    # transpose layout as tensor<2x3xf32>; both represent the same six values.
    if "tensor<2x3xf32>" not in completed.stdout:
        raise RuntimeError(
            "linear-attention plan did not form the six-element K^T V intermediate"
        )
    return completed.stdout


def run_checked(command, *, input_text=None, cwd=None):
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    if completed.returncode != 0:
        rendered_command = " ".join(str(argument) for argument in command)
        raise RuntimeError(
            f"command failed ({rendered_command}):\n"
            + completed.stdout
            + completed.stderr
        )
    return completed.stdout


def lower_and_translate(transformed, module_root, lapis_opt):
    lowering_pipeline = (
        "--sparse-compiler-kokkos="
        "parallelization-strategy=any-storage-any-loop "
        "decompose-sparse-tensors"
    )
    lowered = run_checked(
        [lapis_opt, lowering_pipeline], input_text=transformed
    )
    source = module_root / "algebraic_kernel_fusion_numerical_module.cpp"
    python_wrapper = module_root / "algebraic_kernel_fusion_numerical.py"
    run_checked(
        [
            shutil.which("lapis-translate"),
            "-o",
            source,
            f"--py={python_wrapper}",
            "--finalize",
        ],
        input_text=lowered,
    )
    generated = source.read_text()
    if "float[4][6]" in generated or "float [4][6]" in generated:
        raise RuntimeError(
            "linear-attention lowering retained the quadratic score buffer"
        )
    attention_kernel_signatures = [
        line
        for line in generated.splitlines()
        if line.startswith("void linear_attention_core__lapis_algebraic_kernel_")
    ]
    if not attention_kernel_signatures or any(
        "DualView<float[2][3]" in line for line in attention_kernel_signatures
    ):
        raise RuntimeError(
            "linear-attention K^T V storage was not internalized"
        )
    fused_dot_name = "void fused_dot__lapis_algebraic_kernel_"
    fused_dot_start = generated.find(fused_dot_name)
    fused_dot_definition = generated.find(
        fused_dot_name, fused_dot_start + len(fused_dot_name)
    )
    fused_dot_body_start = generated.find(") {", fused_dot_definition)
    fused_dot_end = generated.find("\n}\n", fused_dot_body_start)
    if min(
        fused_dot_start,
        fused_dot_definition,
        fused_dot_body_start,
        fused_dot_end,
    ) < 0:
        raise RuntimeError("could not find generated fused-dot kernel")
    fused_dot_body = generated[fused_dot_body_start:fused_dot_end]
    if "Kokkos::View<float[4]" in fused_dot_body:
        raise RuntimeError("fused-dot intermediate storage was not eliminated")


def write_native_driver(module_root):
    (module_root / "driver.cpp").write_text(DRIVER_SOURCE)

    kokkos_root = Path(os.environ["KOKKOS_ROOT"])
    kokkos_config = kokkos_root / "lib" / "cmake" / "Kokkos"
    if not kokkos_config.is_dir():
        kokkos_config = kokkos_root / "lib64" / "cmake" / "Kokkos"
    if not (kokkos_config / "KokkosConfig.cmake").is_file():
        raise RuntimeError(f"KokkosConfig.cmake not found under {kokkos_root}")

    support_library = Path(os.environ["SUPPORT_LIB"])
    if not support_library.is_file():
        raise RuntimeError(f"SUPPORT_LIB does not exist: {support_library}")

    cmake_source = f"""\
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(AlgebraicKernelFusionNumerical LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos REQUIRED PATHS \"{kokkos_config}\" NO_DEFAULT_PATH)
add_library(algebraic_kernel_fusion_numerical_module SHARED
  algebraic_kernel_fusion_numerical_module.cpp)
target_link_libraries(algebraic_kernel_fusion_numerical_module
  PRIVATE Kokkos::kokkos \"{support_library}\")
add_executable(algebraic_kernel_fusion_numerical driver.cpp)
target_link_libraries(algebraic_kernel_fusion_numerical
  PRIVATE algebraic_kernel_fusion_numerical_module)
"""
    (module_root / "CMakeLists.txt").write_text(cmake_source)


def build_and_run_native_driver(module_root):
    build_directory = module_root / "build"
    run_checked(
        [
            shutil.which("cmake"),
            "-S",
            module_root,
            "-B",
            build_directory,
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    run_checked(
        [shutil.which("cmake"), "--build", build_directory, "--parallel", "2"]
    )
    output = run_checked(
        [build_directory / "algebraic_kernel_fusion_numerical"]
    )
    print(output, end="")


def main():
    args = parse_args()
    skip_if_runtime_is_unavailable()

    transformed = transform_module(args.lapis_opt)
    module_root = args.work_dir / "algebraic_kernel_fusion_numerical"
    if module_root.exists():
        shutil.rmtree(module_root)
    module_root.mkdir(parents=True)
    lower_and_translate(transformed, module_root, args.lapis_opt)
    write_native_driver(module_root)
    build_and_run_native_driver(module_root)


if __name__ == "__main__":
    main()
