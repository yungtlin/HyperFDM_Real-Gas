###########
# Modules #
###########
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../libs/real-gas/")
from gas_models import get_model_RG8


if __name__ == "__main__":
    model = get_model_RG8()
    # N2: 0.79, O2:0.21
    model.init_composition([0.79, 0.21, 0, 0, 0, 0, 0, 0])

    p_atm_array = np.logspace(-4, 2, 7)

    p_array = p_atm_array*101325    
    n_p = p_atm_array.shape[0]

    nT = 148

    T_array = np.linspace(300, 15000, nT)
    Z_table = np.zeros((n_p, nT))

    for p_idx, p in enumerate(p_array):
        for T_idx, T in enumerate(T_array):
            if T_idx == 0:
                p_s0 = []
            else:
                p_s0 = model.p_all
            print("T: %i"%T)
            model.compute_pT(p, T, p_s0)

            Z = model.Z
            ZERT = model.Z*model.e_mix*model.M_hat_mix/(model.R_hat*model.T)
            ZHRT = model.Z*model.h_mix*model.M_hat_mix/(model.R_hat*model.T)
            ZSR = model.Z*model.s_mix*model.M_hat_mix/model.R_hat
            rhoRTp = model.rho_mix*model.R_hat*model.T/p

            Z_table[p_idx, T_idx] = rhoRTp

    print(model.species_list[-1].M_hat)

    fig, ax = plt.subplots(figsize=(10, 6))

    for p_idx in range(n_p):
        ax.plot(T_array, Z_table[p_idx])

    xlim = [0, 15000]
    ylim = [0, 0.05]

    ax.set_xticks(np.linspace(xlim[0], xlim[1], 16))
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 31), minor=True)
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 8))
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 21), minor=True)

    #plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid(which="both")
    plt.show()

