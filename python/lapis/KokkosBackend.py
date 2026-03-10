import ctypes
import numpy as np
import os
import sys
import subprocess
import tempfile
#import torch
from shutil import which

class KokkosBackend:
    """Main entry-point for the Kokkos backend for linalg-on-tensors dense/sparse code."""

    def __init__(self, decompose_tensors = False, parallel_strategy="any-storage-any-loop", dump_mlir = False, index_instance=0, num_instances=0, ws = os.getcwd()):
        super().__init__()
        self.dump_mlir = dump_mlir
        self.ws = ws
        self.index_instance = index_instance
        self.num_instances = num_instances
        self.decompose_tensors = decompose_tensors
        self.parallel_strategy = parallel_strategy
        if self.index_instance == 0:
            self.package_name = "lapis_package"
        else:
            self.package_name = "lapis_package_" + str(self.index_instance)

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
        #print("Generating CMakeLists.txt...")
        cmake = open(moduleRoot + "/CMakeLists.txt", "w")
        cmake.write("cmake_minimum_required(VERSION 3.16 FATAL_ERROR)\n")
        cmake.write("project(" + self.package_name + ")\n")
        cmake.write("find_package(Kokkos REQUIRED\n")
        cmake.write(" PATHS ")
        cmake.write(kokkosLibDir)
        cmake.write("/cmake/Kokkos)\n")
        cmake.write("add_library(" + self.package_name + "_module SHARED " + self.package_name + "_module.cpp)\n")
        cmake.write("target_link_libraries(" + self.package_name + "_module Kokkos::kokkos)\n")
        if linkSparseSupportLib:
            if 'SUPPORT_LIB' in os.environ:
                support_lib = os.environ['SUPPORT_LIB']
            elif 'SUPPORTLIB' in os.environ:
                print("SUPPORTLIB (deprecated) is set as env variable but SUPPORT_LIB is not.")
                print("SUPPORTLIB is will be used for now, but please switch your environment to set SUPPORT_LIB instead")
                support_lib = os.environ['SUPPORTLIB']
            else:
                raise Exception("SUPPORT_LIB must be defined as an environment variable, and be an absolute path to libmlir_c_runner_utils.so")
            cmake.write("target_link_libraries(" + self.package_name + "_module " + support_lib + ")\n")
        cmake.close()
        # Now configure the project and build the shared library from the build dir
        subprocess.run(['cmake', "-DCMAKE_CXX_EXTENSIONS=OFF", "-DCMAKE_BUILD_TYPE=Debug", moduleRoot], cwd=buildDir)
        buildOut = subprocess.run(['make'], cwd=buildDir, shell=True)
        sys.path.insert(0, moduleRoot)
        lapis = __import__(self.package_name)
        return lapis

    def run_cli(self, app, flags, stdin):
        appAbsolute = which(app)
        p = subprocess.Popen([appAbsolute] + flags, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        (out, errs) = p.communicate(input=stdin)
        if p.returncode != 0:
            print(errs)
            raise Exception("CLI utility failed\n")
        return out

    def compile(self, module):
        moduleText = str(module)
        if self.dump_mlir:
            print("== High-Level ==")
            print(moduleText)

        moduleRoot = self.ws + "/" + self.package_name
        os.makedirs(moduleRoot, exist_ok=True)
        cppOut = moduleRoot + "/" + self.package_name + "_module.cpp"
        pyOut = moduleRoot + "/" + self.package_name + ".py"

        # First lower to Kokkos dialect
        par = self.parallel_strategy
        dst = ""
        if self.decompose_tensors:
            pipeline = f'--sparse-compiler-kokkos=parallelization-strategy={par} decompose-sparse-tensors'
        else:
            pipeline = f'--sparse-compiler-kokkos=parallelization-strategy={par}'
        moduleLowered = ""
        try:
            moduleLowered = self.run_cli("lapis-opt", [pipeline], moduleText)
        except:
            raise Exception("Lowering to Kokkos dialect failed.")

        if self.dump_mlir:
            print("== Lowered ==")
            print(moduleLowered)

        # Then emit C++
        args = ["-o", cppOut, "--py=" + pyOut]
        if self.num_instances == 0 or (self.index_instance == self.num_instances - 1):
            args.append("--finalize")
        try:
            self.run_cli("lapis-translate", args, moduleLowered)
        except:
            raise Exception("Emitting Kokkos C++ failed.")

        # And compile + load the module
        return self.compile_kokkos_to_native(moduleRoot, True)

    def validate_activities(self, activities):
        pass
        #for a in activities:
        #    if a not in ['out', 'dup', 'dupnoneed', 'const', 'inactive']:
        #        raise Exception("Invalid argument activity: each must be one of 'out', 'dup', 'dupnoneed', 'const', 'inactive'.")

    def forward_diff_compile(self, module, fn, d_fn, returnActivities, argActivities):
        # Make sure $ENZYME_OPT is defined
        if 'ENZYME_OPT' not in os.environ:
            raise Exception("AD requires $ENZYME_OPT to be the absolute path to enzymemlir-opt")

        # First, lower the module to SCF
        moduleText = str(module)
        if self.dump_mlir:
            print("== High-Level ==")
            print(moduleText)

        moduleRoot = self.ws + "/" + self.package_name
        os.makedirs(moduleRoot, exist_ok=True)
        cppOut = moduleRoot + "/" + self.package_name + "_module.cpp"
        pyOut = moduleRoot + "/" + self.package_name + ".py"

        # First lower to Kokkos dialect
        par = self.parallel_strategy
        dst = ""
        if self.decompose_tensors:
            pipeline = f'--sparse-compiler-kokkos-pre-ad=parallelization-strategy={par} decompose-sparse-tensors'
        else:
            pipeline = f'--sparse-compiler-kokkos-pre-ad=parallelization-strategy={par}'
        try:
            moduleText = self.run_cli("lapis-opt", [pipeline], moduleText)
        except:
            raise Exception("Pre-AD lowering pipeline failed.")

        if self.dump_mlir:
            print("== Lowered (Pre-AD) ==")
            print(moduleText)

        # Generate derivative function declarations
        self.validate_activities(returnActivities)
        self.validate_activities(argActivities)
        retTys = ','.join(['enzyme_' + a for a in returnActivities])
        argTys = ','.join(['enzyme_' + a for a in argActivities])
        enzymeWrapArgs = f'--enzyme-wrap=infn={fn} outfn={d_fn} retTys={retTys} argTys={argTys} mode=ForwardMode'
        try:
            moduleText = self.run_cli(os.environ['ENZYME_OPT'], [enzymeWrapArgs, '--canonicalize', '--remove-unnecessary-enzyme-ops', '--canonicalize', '--enzyme-simplify-math'], moduleText)
        except:
            raise Exception("Failed to perform Enzyme forward mode differentiation.")
        if self.dump_mlir:
            print("== After forward mode AD: ==")
            print(moduleText)
        # Finish lowering to Kokkos
        try:
            moduleText = self.run_cli("lapis-opt", ['--sparse-compiler-kokkos-post-ad'], moduleText)
        except:
            raise Exception("Lowering to Kokkos dialect failed.")
        if self.dump_mlir:
            print("== After lowering to Kokkos dialect: ==")
            print(moduleText)

        # Then emit C++
        args = ["-o", cppOut, "--py=" + pyOut]
        if self.num_instances == 0 or (self.index_instance == self.num_instances - 1):
            args.append("--finalize")
        try:
            self.run_cli("lapis-translate", args, moduleText)
        except:
            raise Exception("Emitting Kokkos C++ failed.")

        # And compile + load the module
        return self.compile_kokkos_to_native(moduleRoot, True)

    def reverse_diff_compile(self, module, fn, d_fn, returnActivities, argActivities):
        # Make sure $ENZYME_OPT is defined
        if 'ENZYME_OPT' not in os.environ:
            raise Exception("AD requires $ENZYME_OPT to be the absolute path to enzymemlir-opt")

        # First, lower the module to SCF
        moduleText = str(module)
        if self.dump_mlir:
            print("== High-Level ==")
            print(moduleText)

        moduleRoot = self.ws + "/" + self.package_name
        os.makedirs(moduleRoot, exist_ok=True)
        cppOut = moduleRoot + "/" + self.package_name + "_module.cpp"
        pyOut = moduleRoot + "/" + self.package_name + ".py"

        # First lower to Kokkos dialect
        par = self.parallel_strategy
        dst = ""
        if self.decompose_tensors:
            pipeline = f'--sparse-compiler-kokkos-pre-ad=parallelization-strategy={par} decompose-sparse-tensors'
        else:
            pipeline = f'--sparse-compiler-kokkos-pre-ad=parallelization-strategy={par}'
        try:
            moduleText = self.run_cli("lapis-opt", [pipeline], moduleText)
        except:
            raise Exception("Pre-AD lowering pipeline failed.")

        if self.dump_mlir:
            print("== Lowered (Pre-AD) ==")
            print(moduleText)

        # Generate derivative function declarations
        self.validate_activities(returnActivities)
        self.validate_activities(argActivities)
        retTys = ','.join(['enzyme_' + a for a in returnActivities])
        argTys = ','.join(['enzyme_' + a for a in argActivities])
        enzymeWrapArgs = f'--enzyme-wrap=infn={fn} outfn={d_fn} retTys={retTys} argTys={argTys} mode=ReverseModeCombined'
        try:
            print("Full command:", os.environ['ENZYME_OPT'], '"' + enzymeWrapArgs + '"', '--canonicalize', '--remove-unnecessary-enzyme-ops', '--canonicalize', '--enzyme-simplify-math')
            moduleText = self.run_cli(os.environ['ENZYME_OPT'], [enzymeWrapArgs, '--canonicalize', '--remove-unnecessary-enzyme-ops', '--canonicalize', '--enzyme-simplify-math'], moduleText)
        except:
            raise Exception("Failed to perform Enzyme reverse mode differentiation.")
        if self.dump_mlir:
            print("== After reverse mode AD: ==")
            print(moduleText)
        # Finish lowering to Kokkos
        try:
            moduleText = self.run_cli("lapis-opt", ['--sparse-compiler-kokkos-post-ad'], moduleText)
        except:
            raise Exception("Lowering to Kokkos dialect failed.")
        if self.dump_mlir:
            print("== After lowering to Kokkos dialect: ==")
            print(moduleText)

        # Then emit C++
        args = ["-o", cppOut, "--py=" + pyOut]
        if self.num_instances == 0 or (self.index_instance == self.num_instances - 1):
            args.append("--finalize")
        try:
            self.run_cli("lapis-translate", args, moduleText)
        except:
            raise Exception("Emitting Kokkos C++ failed.")

        # And compile + load the module
        return self.compile_kokkos_to_native(moduleRoot, True)
