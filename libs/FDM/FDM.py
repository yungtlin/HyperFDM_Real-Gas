# Subprogram: Finite Difference Coefficient Calculator (for all problems)
#
# Author: Yung-Tien Lin (UCLA)
# Created: April, 19, 2021
# Updated: April, 19, 2021


###########
# Modules #
###########
import numpy as np 

###########
# SubFunc #
###########
# the form is defined as [N-N0-M-M0] see Zhong 1998 for the details
# b*f^{n}_i - a*(dx^-n)f_i = 0
def Zhong(order, form, loc, alpha=0, multiplier=1):
    # error detection
    N, N0, M, M0 = form
    
    # Input indices check
    taylor_buff_size = 5
    loc_sorted_arg = np.argsort(loc)
    loc_sorted = np.array(np.array(loc)[loc_sorted_arg])
    if loc_sorted[0] > 0 or loc_sorted[-1] < 0:
        raise ValueError("Input Locations Must Satify: min <= 0 <= max")

    idx_0 = np.where(np.abs(loc_sorted) < 1e-14)[0][0]
    h = np.mean(loc[1:] - loc[:-1])

    loc_size = loc_sorted.shape[0]

    if N > loc_size or M > loc_size:
        raise ValueError("Zhong Method: N > loc_size or M > loc_size")

    array_size = N + M
    if array_size < (order + 1):
        raise ValueError("Insufficient Amount of Stencils for Solving Matrix")

    # Create the Taylor expansion matrix with some buffer for accuracy evaluation
    taylor_size = array_size + taylor_buff_size
    U = np.zeros((array_size, taylor_size))

    # Start generate U matrix
    idx = 0

    # constant part
    Nr = N0
    Nl = -(N - Nr - 1)
    for i in range(Nl, Nr+1):
        dist = loc_sorted[i + idx_0]
        U[idx, :] = -taylor_exp(taylor_size, dist, 0)
        idx += 1

    # derivative part
    Mr = M0
    Ml = -(M - Mr - 1)
    for i in range(Ml, Mr+1):
        dist = loc_sorted[i + idx_0]
        U[idx, :] = taylor_exp(taylor_size, dist, order)
        idx += 1

    # Generating A, b matrices
    A = np.array(U[:, 0:array_size].T) 
    b = np.zeros((array_size, 1))

    # forcing b_0 equal to the given multiplier value for having a non-trivial solution
    idx_b0 = N - Ml 
    A[-1, :] = 0
    A[-1, idx_b0] = 1
    b[-1] = 1 * multiplier

    # alpha definition
    p = N + M - 2
    b[p] = - alpha/np.math.factorial(p)*(h**(p-order)) * multiplier

    coeff = np.linalg.solve(A, b)
    coeff_a = coeff[:N].reshape(-1)
    coeff_b = coeff[N:].reshape(-1)
    
    return coeff_a, coeff_b

# MLC formulation
# f^n_i = a*f_i + b*f^1_i
def MLC(order, form, loc, alpha=0):

    # Error detection
    L1, L2, M1, M2 = form

    if order <= 1:
        raise ValueError("MLC order must > 1")

    # Input indices check
    taylor_buff_size = 5
    loc_sorted_arg = np.argsort(loc)
    loc_sorted = np.array(np.array(loc)[loc_sorted_arg])
    if loc_sorted[0] > 0 or loc_sorted[-1] < 0:
        raise ValueError("Input Locations Must Satify: min <= 0 <= max")

    idx_0 = np.where(np.abs(loc_sorted) < 1e-14)[0][0]
    h = np.mean(loc[1:] - loc[:-1])

    loc_size = loc_sorted.shape[0]
    L = L1 + L2 + 1
    M = M1 + M2 + 1
    if L > loc_size or M > loc_size:
        raise ValueError("Error: L > loc_size or M > loc_size")

    array_size = L + M
    if array_size < (order + 1):
        raise ValueError("Insufficient Amount of Stencils for Solving Matrix")

    # Create the Taylor expansion matrix with some buffer for accuracy evaluation
    taylor_size = array_size + taylor_buff_size
    U = np.zeros((array_size, taylor_size))

    # Start generate U matrix
    idx = 0

    # constant part
    for i in range(-L1, L2+1):
        dist = loc_sorted[i + idx_0]
        U[idx, :] = taylor_exp(taylor_size, dist, 0)
        idx += 1

    # 1st derivative part
    for i in range(-M1, M2+1):
        dist = loc_sorted[i + idx_0]
        U[idx, :] = taylor_exp(taylor_size, dist, 1)
        idx += 1        

    # Generating A, b matrices
    A = np.array(U[:, 0:array_size].T) 
    b = np.zeros((array_size, 1))

    # fix the value for the right order term
    b[order] = 1

    # alpha definition
    p = L + M - 1
    b[p] = alpha/np.math.factorial(p)*(h**(p-order))

    coeff = np.linalg.solve(A, b)
    coeff_a = coeff[:L].reshape(-1)
    coeff_b = coeff[L:].reshape(-1)

    return coeff_a, coeff_b

# Generating Taylor expansion polynomial for FDMCoeff
def taylor_exp(taylor_size, dist, shift):
    coeff = np.zeros((1, taylor_size))
    for idx in range(shift, taylor_size):
        i = idx - shift
        coeff[0, idx] = dist**(i) / np.math.factorial(i)
    return coeff


def compact_inter(order, form_MLC, dx):
    # Form selection
    L1, L2, M1, M2 = form_MLC
    L = L1 + L2 + 1
    M = M1 + M2 + 1

    # Stencil location array (0 is the point of evaluation) 
    y_list = np.arange(-L1, L2+1)*dx
    yx_list = np.arange(-M1, M2+1)*dx

    # Size evaluation
    array_size = L + M
    taylor_buff_size = 5
    taylor_size = array_size + taylor_buff_size
    U = np.zeros((array_size, taylor_size))

    # skip index for removing target from the A matrix
    if order == 0:
        idx_0 = np.argwhere(y_list == 0)[0, 0]
        skip_idx = idx_0
    elif order == 1:
        idx_0 = np.argwhere(yx_list == 0)[0, 0]
        skip_idx = idx_0 + L

    # 
    idx = 0
    # constant part
    for y in y_list:
        U[idx, :] = taylor_exp(taylor_size, y, 0)
        idx += 1

    # 1st derivative part
    for y in yx_list:
        U[idx, :] = taylor_exp(taylor_size, y, 1)
        idx += 1        

    # Generating A, b matrices
    A = np.array(U[:, 0:array_size-1].T) 
    A = np.delete(A, skip_idx, 1) # remove the skip item

    b = np.zeros((array_size-1, 1))
    b[order] = 1

    c = (np.linalg.solve(A, b).T)[0]
    
    coeff_a = c[:L+order-1]
    coeff_b = c[L+order-1:]
    return coeff_a, coeff_b

def compact_extrap(order, form_MLC, dx):
    # Form selection
    L1, L2, M1, M2 = form_MLC
    L = L1 + L2
    M = M1 + M2

    loc_l1 = np.arange(-L1, 0)
    loc_l2 = np.arange(1, L2 + 1)
    loc_l = np.append(loc_l1, loc_l2)*dx

    loc_m1 = np.arange(-M1, 0)
    loc_m2 = np.arange(1, M2 + 1)
    loc_m = np.append(loc_m1, loc_m2)*dx

    array_size = L + M
    A = np.zeros((array_size, array_size))

    # 
    idx = 0
    # constant part
    for y in loc_l:
        A[idx, :] = taylor_exp(array_size, y, 0)
        idx += 1

    # 1st derivative part
    for y in loc_m:
        A[idx, :] = taylor_exp(array_size, y, 1)
        idx += 1        

    A_inv = np.linalg.inv(A)

    coeffs = A_inv[order]

    c_a = coeffs[:L]
    c_b = coeffs[L:]

    return c_a, c_b

def FDM_interp(order, dx_list):
    n_L = len(dx_list)
    if order > n_L - 1:
        raise ValueError("FDM Interpolation Derivative Order Is Too Large")
    
    A = np.zeros((n_L, n_L))
    for dx_idx, dx in enumerate(dx_list):
        A[dx_idx] = taylor_exp(n_L, dx, 0)

    c = np.linalg.inv(A)[order]

    return c


if __name__ == "__main__":
    order = 1
    form = [2, 0, 1, 0]
    L1 = form[0]
    L2 = form[1]
    M1 = form[2]
    M2 = form[3]

    N_list = [4, 8, 16, 32, 64, 128]
    dev_list = []

    order = 1

    for N in N_list:
        x_n = np.pi
        x = np.linspace(0, x_n, N + 1)[:-1]
        #print(x)
        dx = np.mean(x[1:] - x[:-1])

        c_a, c_b = compact_extrap(order, form, dx)

        y0 = np.sin(x)
        y1 = np.cos(x)
        
        #y2_mlc = np.sum(c_a*y0[:form[1]]) + np.sum(c_b*y1[:form[3]])
        y2_mlc = np.sum(c_a*y0[-form[0]:]) + np.sum(c_b*y1[-form[2]:])
        
        #dev = np.abs(np.sin(x_n) - y2_mlc)
        dev = np.abs(np.cos(x_n) - y2_mlc)

        dev_list += [dev]

    dev_arr = np.array(dev_list)

    import matplotlib.pyplot as plt 
    plt.loglog(N_list, dev_list, "-ob")

    r = dev_arr[1:]/dev_arr[:-1]
    m = -np.log(r)/np.log(2)
    print("m:", m)
    plt.show()


