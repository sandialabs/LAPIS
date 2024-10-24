# Initialize LAPIS python extension.
import lapis._mlir_libs._lapis

import ctypes
import numpy as np
import os
import sys
import subprocess
import tempfile
import torch

# For running passes/pipelines on a module
import lapis.ir
import lapis.passmanager

# For emitting a lowered module to Kokkos C++
from lapis._mlir_libs._lapis import emit_kokkos
from lapis._mlir_libs._lapis import emit_kokkos_sparse

LOWERING_PIPELINE = "builtin.module(" + ",".join([
    "func.func(refback-generalize-tensor-pad)",
    "func.func(linalg-fuse-elementwise-ops)",
    "func.func(scf-bufferize)",
    "func.func(tm-tensor-bufferize)",
    "func.func(empty-tensor-to-alloc-tensor)",
    "func.func(linalg-bufferize)",
    "func-bufferize",
    "arith-bufferize",
    "refback-mlprogram-bufferize",
    "func.func(tensor-bufferize)",
    "func.func(finalizing-bufferize)",
    "func.func(buffer-deallocation)",
    "func.func(tm-tensor-to-loops)",
    "func.func(convert-linalg-to-parallel-loops)",
    #"builtin.func(convert-linalg-to-loops)",
    "func.func(lower-affine)",
]) + ")"

### Original RefBackend pipeline for full lowering to LLVM (is more low-level than what we need for Kokkos)
#LOWERING_PIPELINE = ",".join([
#    "builtin.func(refback-generalize-tensor-pad)",
#    # Bufferize.
#    "builtin.func(scf-bufferize)",
#    "builtin.func(tm-tensor-bufferize)",
#    "builtin.func(linalg-bufferize)",
#    "func-bufferize",
#    "arith-bufferize",
#    "builtin.func(tensor-bufferize)",
#    "builtin.func(finalizing-bufferize)",
#    # Munge to make it ExecutionEngine compatible.
#    # Specifically, we rewrite calling convention boundaries to be in terms
#    # of unranked memref, and we rewrite the return to actually be a
#    # callback that consumes the return (the final munged function always
#    # returns void at the C level -- we get the return value by providing the
#    # callback).
#    "refback-munge-calling-conventions",
#    # Insert global variable and instruction sequence for getting the next
#    # global seed used in stateful rng.
#    "refback-insert-rng-globals",
#    # Lower to LLVM
#    "builtin.func(tm-tensor-to-loops)",
#    "builtin.func(refback-munge-memref-copy)",
#    "builtin.func(convert-linalg-to-loops)",
#    "builtin.func(lower-affine)",
#    "convert-scf-to-cf",
#    "builtin.func(refback-expand-ops-for-llvm)",
#    "builtin.func(arith-expand)",
#    "builtin.func(convert-math-to-llvm)",
#    "convert-linalg-to-llvm",
#    "convert-memref-to-llvm",
#    #"builtin.func(convert-arith-to-llvm)",
#    #"convert-func-to-llvm",
#    #"convert-cf-to-llvm",
#    #"reconcile-unrealized-casts",
#])

class KokkosBackend:
    """Main entry-point for the Kokkos LinAlg backend."""

    def __init__(self, dump_mlir = False, before_mlir_filename = "dump.mlir", after_mlir_filename = "after_dump.mlir", index_instance=0, num_instances=0, ws = os.getcwd()):
        super().__init__()
        self.dump_mlir = dump_mlir
        self.before_mlir_filename = before_mlir_filename
        self.after_mlir_filename = after_mlir_filename
        self.ws = ws
        self.index_instance = index_instance
        self.num_instances = num_instances
        if self.index_instance == 0:
            self.package_name = "lapis_package"
        else:
            self.package_name = "lapis_package_"+str(self.index_instance)
        self.context = lapis.ir.Context()

    def compile_kokkos_to_native(self, moduleRoot, linkSparseSupportLib):
        # Now that we have a Kokkos source file, generate the CMake to build it into a shared lib,
        # using $KOKKOS_ROOT as the kokkos installation.
        buildDir = moduleRoot + "/build"
        # First, clean existing CMakeCache.txt from build if it exists
        if os.path.isfile(buildDir + '/CMakeCache.txt'):
            os.remove(buildDir + '/CMakeCache.txt')
        # Create the source and build directories
        os.makedirs(buildDir, exist_ok=True)
        if 'KOKKOS_ROOT' not in os.environ:
            raise Exception("KOKKOS_ROOT must be defined as an environment variable, and point to a Kokkos installation!")
        kokkosDir = os.environ['KOKKOS_ROOT']
        kokkosLibDir = kokkosDir + "/lib"
        if not os.path.isdir(kokkosLibDir):
          kokkosLibDir = kokkosLibDir + "64"
        if not os.path.isfile(kokkosLibDir + "/cmake/Kokkos/KokkosConfig.cmake"):
            raise Exception("Did not find file $KOKKOS_ROOT/lib/cmake/Kokkos/KokkosConfig.cmake or $KOKKOS_ROOT/lib64/cmake/Kokkos/KokkosConfig.cmake. Check Kokkos installation and make sure $KOKKOS_ROOT points to it.")
        print("Generating CMakeLists.txt...")
        cmake = open(moduleRoot + "/CMakeLists.txt", "w")
        cmake.write("project(" + self.package_name + ")\n")
        cmake.write("cmake_minimum_required(VERSION 3.16 FATAL_ERROR)\n")
        cmake.write("find_package(Kokkos REQUIRED\n")
        cmake.write(" PATHS ")
        cmake.write(kokkosLibDir)
        cmake.write("/cmake/Kokkos)\n")
        cmake.write("add_library(" + self.package_name + "_module SHARED " + self.package_name + "_module.cpp)\n")
        cmake.write("target_link_libraries(" + self.package_name + "_module Kokkos::kokkos)\n")
        if linkSparseSupportLib:
            if 'SUPPORTLIB' not in os.environ:
                raise Exception("SUPPORTLIB must be defined as an environment variable, and be an absolute path to libmlir_c_runner_utils.so")
            supportlib = os.environ['SUPPORTLIB']
            cmake.write("target_link_libraries(" + self.package_name + "_module " + supportlib + ")\n")
        cmake.close()
        # Now configure the project and build the shared library from the build dir
        print("Configuring build...")
        subprocess.run(['cmake', "-DCMAKE_BUILD_TYPE=Debug", moduleRoot], cwd=buildDir)
        print("Building module...")
        buildOut = subprocess.run(['make'], cwd=buildDir, shell=True)
        print("Importing module...")
        sys.path.insert(0, moduleRoot)
        lapis = __import__(self.package_name)
        if os.path.isfile(buildDir + "/lib" + self.package_name + "_module.so"):
            return lapis.LAPISModule(buildDir + "/lib" + self.package_name + "_module.so")
        if os.path.isfile(buildDir + "/lib" + self.package_name + "_module.dylib"):
            return lapis.LAPISModule(buildDir + "/lib" + self.package_name + "_module.dylib")

    def compile_sparse(self, module, options: str = ""):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in PyTaco (linalg on sparse tensors) form.

        Args:
          module: The MLIR module generated from pytaco.
        Returns:
          An instance of a wrapper class which has the module's functions as callable methods.
        """

        module_name = "MySparseModule"
        #original_stderr = sys.stderr
        # Lower module in place to make it ready for compiler backends.

        # Convert to string and re-parse, to avoid redefining common dialects in registry
        print("Input module's type:")
        print(type(module))
        print("Input module text:")
        asm = str(module)
        print(asm)
        lapisModule = lapis.ir.Module.parse(asm, context=self.context)
        print("Result module (LAPIS context):")
        print(str(lapisModule))
 
        #module = tensor.get_expression().get_module(tensor, tensor._assignment.indices)
        #asm_for_error_report = module.operation.get_asm(
        #    large_elements_limit=10, enable_debug_info=True)
        #if "kokkos-uses-hierarchical" in options:
        #    useHierarchical = True
        #    options = options.replace("kokkos-uses-hierarchical", "")
        #else:
        #    useHierarchical = False
        pipeline = f'builtin.module(sparse-compiler-kokkos{{{options} parallelization-strategy=any-storage-any-loop}})'
        pm = PassManager.parse(pipeline, context=self.context)
        pm.run(module.operation)
        #if self.dump_mlir:
        #    with open(self.before_mlir_filename, 'w') as f:
        #        f.write(asm_for_error_report)
        #        print("Wrote out MLIR dump to "+self.before_mlir_filename)
        #    asm_for_error_report = module.operation.get_asm(
        #        large_elements_limit=10, enable_debug_info=True)
        #    with open(self.after_mlir_filename, 'w') as f:
        #        f.write(asm_for_error_report)
        #        print("Wrote out ASM to "+self.after_mlir_filename)
        moduleRoot = self.ws + "/" + self.package_name
        os.makedirs(moduleRoot, exist_ok=True)
        # Generate Kokkos C++ source from the module.
        print("Emitting sparse module as Kokkos C++...")
        emit_kokkos_sparse(module, moduleRoot + "/" + self.package_name + "_module.cpp", moduleRoot + "/" + self.package_name + ".py", useHierarchical, self.index_instance==self.num_instances)
        return self.compile_kokkos_to_native(moduleRoot, True)

