# RUN: SUPPORTLIB=%mlir_lib_dir/libmlir_c_runner_utils%shlibext %PYTHON %s | FileCheck %s

import numpy as np
import os
import sys
import tempfile
import shutil
import math
import copy

# Compute determinant of Jacobian for an element given by coordinates (x,y,z)
def compute_det_jac(x, y, z, gp):
    omx = 0.5*(1-gp[0])
    opx = 0.5*(1+gp[0])
    ome = 0.5*(1-gp[1])
    ope = 0.5*(1+gp[1])
    A = np.array([[-ome,  ome, -ope, ope],
                  [-omx, -opx,  omx, opx] ])
    J = 0.5*np.linalg.det(A@np.column_stack((x,y)))
    return J

def compute_spatial_integral(u_all, x_all, y_all, z_all, var, time, func):
    import pyttb as ttb
    import math

    num_elem = u_all.shape[0]
    num_node_per_elem = u_all.shape[1]
    num_qp_per_elem = num_node_per_elem
    num_var = len(var)
    num_time = len(time)
    num_dim = math.log2(num_node_per_elem)
    assert num_dim == 2

    # basis functions evaluated at quadrature points
    oor3 = 1/math.sqrt(3)
    op = 0.5*(1+oor3)
    om = 0.5*(1-oor3)
    A = np.array([[op, om], [om, op]])
    A = np.kron(A,A)
    gp = np.array([ [-oor3, -oor3],
                    [ oor3, -oor3],
                    [-oor3,  oor3],
                    [ oor3,  oor3] ])

    # quadrature weights (all 1 for linear Gaussian quadrature)
    w = np.ones(num_qp_per_elem)

    # determinant of Jacobian of transformation from physical to reference element
    J = np.zeros(num_qp_per_elem)
        
    u = np.zeros((num_node_per_elem, num_var, num_time), order='F')      # variables at nodes
    v = np.zeros((num_qp_per_elem, num_var, num_time), order='F')        # variables at gauss points
    uu = np.reshape(u, (num_node_per_elem, num_var*num_time), order='F') # aliases u as a matrix
    vv = np.reshape(v, (num_qp_per_elem, num_var*num_time), order='F')   # aliases v as a matrix

    p = np.zeros(num_time)
    for e in range(0,num_elem):
        # var fields at nodes
        for i in range(0, num_node_per_elem):
            for j in range(0, num_var):
                for k in range(0, num_time):
                    u[i, j, k] = u_all[e, i, j, k]
        #u = copy.deepcopy(u_all[e, :, :, :])

        # coordinates of each node
        x = x_all[e, :]
        y = y_all[e, :]
        z = z_all[e, :]

        # interpolate variables to the Gauss points
        np.matmul(A, uu, out=vv)

        # compute function at Gauss points
        f = func(v)

        # compute det(J) at each gauss point
        for j in range(0,num_qp_per_elem):
            J[j] = compute_det_jac(x, y, z, gp[j,:])

        # add contribution to integral
        for j in range(num_qp_per_elem):
            p += w[j]*J[j]*f[j,:]

    return p


num_time = 10
num_var = 10
num_elem_x = 10
num_elem_y = 10
num_elem = num_elem_x * num_elem_y
num_gp = 4
num_node_per_elem = 4

var = np.arange(0, num_var)
time = np.linspace(0, 1., num_time)

u_all = np.ones((num_elem, num_node_per_elem, num_var, num_time), order='F')      # variables at nodes
x_all = np.zeros((num_elem, num_node_per_elem), order='F')
y_all = np.zeros((num_elem, num_node_per_elem), order='F')
z_all = np.zeros((num_elem, num_node_per_elem), order='F')

dx = 1
dy = 1

for i in range(0, num_elem_x):
    for j in range(0, num_elem_y):
        element_id = num_elem_x * j + i

        x_all[element_id, 0] = i * dx
        y_all[element_id, 0] = j * dy
        x_all[element_id, 1] = (i+1) * dx
        y_all[element_id, 1] = j * dy
        x_all[element_id, 2] = i * dx
        y_all[element_id, 2] = (j+1) * dy
        x_all[element_id, 3] = (i+1) * dx
        y_all[element_id, 3] = (j+1) * dy


func = lambda v: np.sum((v-3)**2,axis=1)

p = compute_spatial_integral(u_all, x_all, y_all, z_all, var, time, func)

print(p)