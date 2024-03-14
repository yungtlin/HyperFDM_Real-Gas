###########
# Modules #
###########
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../libs/real-gas/")
from Species import Monatomic, Diatomic
from Mixture import Reaction, Mixture, get_RG8_a_jacobian
from gas_models import get_model_RG8


def res_a(U, mixture):
    eta_all = U[:8]
    rho = U[-2]
    T = U[-1]

    K_p_all = mixture.compute_K_p(T)
    ratio_NO = mixture.ratio_NO
    R_hat = mixture.R_hat
    n_s = mixture.n_species

    rhoRT = rho*R_hat*T

    res = np.zeros(10)

    res[0] =   eta_all[3]**2/eta_all[0] - K_p_all[0]/rhoRT
    res[1] =   eta_all[4]**2/eta_all[1] - K_p_all[1]/rhoRT
    res[2] = eta_all[3]*eta_all[4]/eta_all[2] - K_p_all[2]/rhoRT
    res[3] = eta_all[5]*eta_all[7]/eta_all[3] - K_p_all[3]/rhoRT
    res[4] = eta_all[6]*eta_all[7]/eta_all[4] - K_p_all[4]/rhoRT

    res[5] = (2*eta_all[0] + eta_all[2] + eta_all[3] + eta_all[5])/\
        (2*eta_all[1] + eta_all[2] + eta_all[4] + eta_all[6]) - ratio_NO

    res[6] = eta_all[5] + eta_all[6] - eta_all[7]


    mixture.set_T(T)

    for s_idx in range(n_s):
        species = mixture.species_list[s_idx]
        p_s = eta_all[s_idx]*rho*R_hat*T
        species.compute_entropy(p_s)
        s_total = species.s_total

        cs = eta_all[s_idx]*species.M_hat*s_total

        res[7] += species.M_hat*eta_all[s_idx]
        res[8] += p_s
        res[9] += cs

    return res

if __name__ == "__main__":
    # initialize gas model
    model = get_model_RG8()
    # N2: 0.79, O2:0.21
    model.init_composition([0.79, 0.21, 0, 0, 0, 0, 0, 0])

    # load simulation results
    file = "RG8-results.dat"
    data_RG8 = np.load(file, mmap_mode="r")
    n_p, n_T, n_U = data_RG8.shape

    p_idx = 0
    T_idx = 10

    data_info = data_RG8[p_idx, T_idx]

    p = data_info[0]
    T = data_info[1]
    p_all = data_info[2:]
    model.update_p_all(T, p_all)
    J_exact = get_RG8_a_jacobian(model)


    model.update_p_all(T, p_all)
    rho = model.rho_mix
    T = model.T
    eta_all = model.eta_all

    n_Y = 10
    U0 = np.zeros(n_Y)
    U0[:8] = eta_all
    U0[-2] = rho
    U0[-1] = T

    res_0 = res_a(U0, model)

    dx = 1e-6
    J_FDM = np.zeros((n_Y, n_Y))
    for u_idx in range(n_Y):
        U1 = np.array(U0)
        U2 = np.array(U0)
        dy = U0[u_idx]*dx

        U1[u_idx] += dy
        res_1 = res_a(U1, model)
        
        U2[u_idx] -= dy
        res_2 = res_a(U2, model)

        dres_dx = (res_1 - res_2)/(2*dy)


        J_FDM[:, u_idx] = dres_dx

    u_idx = 9

    dev = J_FDM[u_idx] - J_exact[u_idx]
    y = J_exact[u_idx]
    ref = np.where(np.abs(y) > 1e-20, np.abs(y), 1e-20)
    err_ref = dev/ref

    print(err_ref)
    print()
    #print(np.max(np.abs(err_ref)))
    print(J_exact[u_idx])
    print(J_FDM[u_idx])
    print()
