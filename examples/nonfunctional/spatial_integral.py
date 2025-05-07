# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys
import tempfile
import shutil
import math

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from lapis.tools import mlir_pytaco_api as pt
from lapis.tools import testing_utils as utils

###### This PyTACO part is taken from the TACO open-source project. ######
# See http://tensor-compiler.org/docs/scientific_computing/index.html.

def get_A_gp_w():
    oor3 = 1/math.sqrt(3)
    op = 0.5*(1+oor3)
    om = 0.5*(1-oor3)

    opp = op*op
    omm = om*om
    opm = op*om

    A = pt.tensor([4, 4], [pt.dense, pt.dense], dtype=pt.float64)

    A.insert([0,0], opp)
    A.insert([0,1], opm)
    A.insert([0,2], opm)
    A.insert([0,3], omm)

    A.insert([1,0], opm)
    A.insert([1,1], opp)
    A.insert([1,2], omm)
    A.insert([1,3], opm)

    A.insert([2,0], opm)
    A.insert([2,1], omm)
    A.insert([2,2], opp)
    A.insert([2,3], opm)

    A.insert([3,0], omm)
    A.insert([3,1], opm)
    A.insert([3,2], opm)
    A.insert([3,3], opp)

    gp = pt.tensor([4, 2], [pt.dense, pt.dense], dtype=pt.float64)

    gp.insert([0,0], -oor3)
    gp.insert([0,1], -oor3)

    gp.insert([1,0],  oor3)
    gp.insert([1,1], -oor3)

    gp.insert([2,0], -oor3)
    gp.insert([2,1],  oor3)

    gp.insert([3,0], oor3)
    gp.insert([3,1], oor3)

    w = pt.tensor([4], [pt.dense], dtype=pt.float64)

    for i in range(0, 4):
        w.insert([i], 1.)

    return A, gp, w

# interpolate variables to the Gauss points
def interpolate(A: pt.tensor, u: pt.tensor) -> pt.tensor:
    v = pt.tensor([u.shape[0], u.shape[1], u.shape[2], u.shape[3]], [pt.dense, pt.dense, pt.dense, pt.dense], dtype=pt.float64)
    i, j, k, l, m = pt.get_index_vars(5)

    v[i, j, k, l] = A[j, m] * u[i, m, k, l]
    return v

def func_at_gp(val):
    S = pt.tensor(3, dtype=pt.float64)
    return (val-S[0])*(val-S[0])

# compute function at Gauss points
def func_at_gps(v: pt.tensor) -> pt.tensor:
    f = pt.tensor([v.shape[0], v.shape[1], v.shape[2]], [pt.dense, pt.dense, pt.dense], dtype=pt.float64)

    i, j, k, l = pt.get_index_vars(4)
    f[i, j, l] = func_at_gp(v[i, j, k, l])
    return f

# add contribution to integral
def integrate(w: pt.tensor, J: pt.tensor, f:pt.tensor) -> pt.tensor:
    p = pt.tensor([f.shape[2]], [pt.dense], dtype=pt.float64)
    
    i, j, k = pt.get_index_vars(3)
    p[k] = w[j] * J[i, j] * f[i, j, k]
    return p

def get_mesh_info(num_var = 1, num_time = 1, num_elements_x = 1, num_elements_y = 1):
    num_elements = num_elements_x * num_elements_y

    num_gp = 4
    num_nodes = 4

    #x = pt.tensor([num_elements, num_nodes], [pt.dense, pt.dense], dtype=pt.float64)
    #y = pt.tensor([num_elements, num_nodes], [pt.dense, pt.dense], dtype=pt.float64)
    #z = pt.tensor([num_elements, num_nodes], [pt.dense, pt.dense], dtype=pt.float64)

    u = pt.tensor([num_elements, num_nodes, num_var, num_time], [pt.dense, pt.dense, pt.dense, pt.dense], dtype=pt.float64)

    J = pt.tensor([num_elements, num_gp], [pt.dense, pt.dense], dtype=pt.float64)


    for i in range(0, num_elements_x):
        for j in range(0, num_elements_y):
            element_id = num_elements_x * j + i

            for i_gp in range(0, num_gp):
                J.insert([element_id, i_gp], 0.5)
            for i_node in range(0, num_nodes):
                for i_var in range(0, num_var):
                    for i_time in range(0, num_time):
                        u.insert([element_id, i_node, i_var, i_time], 1.)
    return J, u

num_time = 10
num_var = 10
num_elements_x = 10
num_elements_y = 10
num_elements = num_elements_x * num_elements_y
num_gp = 4

options = "parallelization-strategy=any-storage-any-loop kokkos-uses-hierarchical"

use_function_calls = True
use_one_expression = False
use_Kokkos_backend = True

A, gp, w = get_A_gp_w()
J, u = get_mesh_info(num_var = num_var, num_time = num_time, num_elements_x = num_elements_x, num_elements_y = num_elements_y)

if use_function_calls:
    v = interpolate(A, u)
    f = func_at_gps(v)
    p = integrate(w, J, f)

    if not use_Kokkos_backend:
        v._sync_value()
        f._sync_value()
    else:
        with tempfile.TemporaryDirectory() as test_dir:
            v.compile_kokkos(True, index_instance=0, num_instances=2, options=options)
            v.compute_kokkos()
            f.compile_kokkos(True, index_instance=1, num_instances=2, options=options)
            f.compute_kokkos()

else:
    p = pt.tensor([num_time], [pt.dense], dtype=pt.float64)

    if use_one_expression:
        i, j, k, l, m = pt.get_index_vars(5) 

        #p[l] = w[j] * J[i, j] * func_at_gp(A[j, m] * u[i, m, k, l]) #-> Segmentation fault (core dumped) if use_Kokkos_backend
        p[l] = w[j] * J[i, j] * (A[j, m] * u[i, m, k, l]) * (A[j, m] * u[i, m, k, l])
    else:
        v = pt.tensor([num_elements, num_gp, num_var, num_time], [pt.dense, pt.dense, pt.dense, pt.dense], dtype=pt.float64)
        f = pt.tensor([num_elements, num_gp, num_time], [pt.dense, pt.dense, pt.dense], dtype=pt.float64)

        i, j, k, l, m = pt.get_index_vars(5) 

        v[i, j, k, l] = A[j, m] * u[i, m, k, l]
        f[i, j, l] = func_at_gp(v[i, j, k, l])
        p[l] = w[j] * J[i, j] * f[i, j, l]

        if not use_Kokkos_backend:
            v._sync_value()
            f._sync_value()
        else:
            with tempfile.TemporaryDirectory() as test_dir:
                v.compile_kokkos(True, index_instance=0, num_instances=2, options=options)
                v.compute_kokkos()
                f.compile_kokkos(True, index_instance=1, num_instances=2, options=options)
                f.compute_kokkos()

if use_Kokkos_backend:
    with tempfile.TemporaryDirectory() as test_dir:
        print("Compiling, running spatial integral and writing result to p.tns")
        pt.write_kokkos("p.tns", p, options, index_instance=2, num_instances=2)
else:
    coords, values = p.get_coordinates_and_values()
    print(values)
