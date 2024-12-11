import ctypes
import numpy as np
import os
import sys
import subprocess
import tempfile
import torch
from shutil import which

class KokkosBackend:
    """Main entry-point for the Kokkos LinAlg backend."""

    def __init__(self, dump_mlir = False, index_instance=0, num_instances=0, ws = os.getcwd()):
        super().__init__()
        self.dump_mlir = dump_mlir
        self.ws = ws
        self.index_instance = index_instance
        self.num_instances = num_instances
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

    def compile(self, module, options: str = ""):
        moduleText = str(module)
 
        moduleRoot = self.ws + "/" + self.package_name
        os.makedirs(moduleRoot, exist_ok=True)
        print("Emitting sparse module as Kokkos C++...")
        lapisEmit = which('lapis-emit')
        cppOut = moduleRoot + "/" + self.package_name + "_module.cpp"
        pyOut = moduleRoot + "/" + self.package_name + ".py"
        args = ["--cxx=" + cppOut, "--py=" + pyOut]
        if self.dump_mlir:
            args.append("--dump")
        if self.num_instances == 0 or (self.index_instance == self.num_instances - 1):
            args.append("--final")
        p = subprocess.Popen([lapisEmit] + args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        errs = p.communicate(input=moduleText)[1]
        if p.returncode != 0:
            raise Exception("lapis-emit failed to process module:\n" + errs)
        return self.compile_kokkos_to_native(moduleRoot, True)

