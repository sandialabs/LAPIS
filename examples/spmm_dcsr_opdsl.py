# RUN: env SUPPORT_LIB=%mlir_c_runner_utils \
# RUN:   %PYTHON %s | FileCheck %s

import ctypes
import inspect
from itertools import chain, combinations
import numpy as np
import os
import sys
import time
import multiprocessing as mp

from mlir import ir
from mlir import runtime as rt

from mlir.dialects import sparse_tensor as st
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects import arith
from mlir.dialects import linalg
from mlir.dialects.linalg.opdsl import lang as dsl
from utils.NewSparseTensorFactory import newSparseTensorFactory

from lapis import KokkosBackend

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

def build_iterator_types(*args):
    #return ir.AttrBuilder.get('IteratorTypeArrayAttr')(args)
    return ir.ArrayAttr.get([ir.Attribute.parse(f"#linalg.iterator_type<{x}>") for x in args])

def to_affine_map_attr(source, *subsets):
    ret = []
    for subset in subsets:
        exprs = [ir.AffineExpr.get_dim(source.index(d)) for d in subset]
        ret.append(ir.AffineMapAttr.get(ir.AffineMap.get(len(source), 0, exprs)))
    return ir.ArrayAttr.get(ret)

@dsl.linalg_structured_op
def matmul_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K),
    B=dsl.TensorDef(dsl.T, dsl.S.K, dsl.S.N),
    C=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N, output=True),
):
    C[dsl.D.m, dsl.D.n] += linalg.BinaryFn.mul(A[dsl.D.m, dsl.D.k], B[dsl.D.k, dsl.D.n])

def program_code(attr: st.EncodingAttr):
    module = ir.Module.create()
    f64 = ir.F64Type.get()
    index = ir.IndexType.get()
    NODIM = ir.RankedTensorType.get_dynamic_size()
    
    out = ir.RankedTensorType.get([3,4], f64, attr)
    p1 = ir.RankedTensorType.get([NODIM], index)
    i1 = ir.RankedTensorType.get([NODIM], index)
    p2 = ir.RankedTensorType.get([NODIM], index)
    i2 = ir.RankedTensorType.get([NODIM], index)
    values = ir.RankedTensorType.get([NODIM], f64)

    a_type = ir.RankedTensorType.get([3,4], f64, attr)
    b_type = ir.RankedTensorType.get([4, NODIM], f64)
    c_type = ir.RankedTensorType.get([3, NODIM], f64)
    with ir.InsertionPoint(module.body):
        @func.FuncOp.from_py_func(a_type)
        def print_tensor(a):
            st.print_(a)

        @func.FuncOp.from_py_func(a_type, b_type, c_type)
        def spMxM(a, b, c):
            cf0 = arith.ConstantOp(f64, 0.0)

            mulop = linalg.GenericOp(
                [c_type],
                [a, b],
                [c],
                to_affine_map_attr("ijk", "ji", "ik", "jk"),
                build_iterator_types("reduction", "parallel", "parallel")
            )

            body = mulop.regions[0].blocks.append(f64, f64, f64)
            with ir.InsertionPoint(body):
                a, b, o = body.arguments

                # Multiply Operations
                res = st.binary(f64, a, b)
                overlap = res.owner.regions[0].blocks.append(f64, f64)
                with ir.InsertionPoint(overlap):
                    a, b = overlap.arguments
                    st.yield_([arith.mulf(a, b)])

                # Addition Operations
                res = st.reduce(res, o, cf0)
                reduction = res.owner.regions[0].blocks.append(f64, f64)
                with ir.InsertionPoint(reduction):
                    a,b = reduction.arguments
                    ret = arith.addf(a,b)
                    st.yield_([ret])
                linalg.yield_([res])

            return mulop.result

        #@func.FuncOp.from_py_func(p1, i1, p2, i2, values)
        #def assembler(p1, i1, p2, i2, values):
        #    return st.assemble(a_type, (p1, i1, p2, i2), values)

        #@func.FuncOp.from_py_func(p1, i1, p2, i2, values, b_type, c_type)
        #def alt_main(p1, i1, p2, i2, values, b, c):
        #    A = assembler(p1, i1, p2, i2, values)
        #    return spMxM(A, b, c)


    spMxM.func_op.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
    print_tensor.func_op.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
    #assembler.func_op.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
    #alt_main.func_op.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()

    func_code = "\n".join([str(op) for op in module.operation.regions[0].blocks[0].operations])

    return func_code, module

def build_compile_and_run_SpMM(attr: st.EncodingAttr):
    # Build.

    code, module = program_code(attr)
    NL='\n'
    print("-" * 80)
    print(f"{NL.join(f'{it+1} {line}' for it, line in enumerate(code.split(NL)))}")
    print("-" * 80)

    # Compile.
    # Note: sparsifier generates incorrect code (data race) with any-storage-any-loop.
    # On GPU backends this reliably produces wrong results. Changing c update to atomic fixes this
    # but lowering never uses atomics.
    backend = KokkosBackend.KokkosBackend(dump_mlir=False, parallel_strategy="any-storage-outer-loop")
    engine = backend.compile(module)

    # 2-D sparse tensor A in compressed:compressed format
    p1 = np.array([0, 3], dtype=np.int64)
    i1 = np.array([0, 1, 1], dtype=np.int64)
    p2 = np.array([0, 1, 3, 3], dtype=np.int64)
    i2 = np.array([0, 1, 2], dtype=np.int64)
    values = np.array([1.1, 2.2, 3.3], dtype=np.float64)

    # Dense matrix b
    b = np.array(
        [
            [1.0, 2.0],
            [4.0, 3.0],
            [5.0, 6.0],
            [8.0, 7.0]
        ], dtype=np.float64
    )

    # Output matrix (written by MLIR)
    c = np.zeros((3, 2), np.float64)

    # Use newSparseTensor c interface to create the sparse tensor
    A = newSparseTensorFactory()((3,4), ctypes.c_double, buffers=[p1, i1, p2, i2, values])

    engine.print_tensor(A)
    engine.spMxM(A, b, c)

    # 2-D tensor A in dense format
    a = np.array(
        [
            [1.1, 0.0, 0.0, 0.0],
            [0.0, 2.2, 3.3, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float64
    )
    expected = np.matmul(a, b)

    print("Result c:")
    print(c)

    print("Expected result:")
    print(expected)

    if np.allclose(c, expected):
        print("Success")
        return True
    else:
        print("Failure: result differed significantly from np.matmul")
        return False

def main():
    support_lib = os.getenv("SUPPORT_LIB")
    assert support_lib is not None, "SUPPORT_LIB is undefined"
    if not os.path.exists(support_lib):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), support_lib)

    with ir.Context() as ctx, ir.Location.unknown():
        opt = f"parallelization-strategy=none"
        builder = st.EncodingAttr.build_level_type
        fmt = st.LevelFormat
        prop = st.LevelProperty

        level = [builder(fmt.compressed), builder(fmt.compressed)]
        ordering = ir.AffineMap.get_permutation([0, 1])
        pwidth, iwidth = 0, 0
        attr = st.EncodingAttr.get(
            level, ordering, ordering, pwidth, iwidth
        )

        start_time = time.time()
        # Returns True for success
        sys.exit(0 if build_compile_and_run_SpMM(attr) else 1)

if __name__ == "__main__":
    main()
