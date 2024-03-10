###########
# Modules #
###########
import numpy as np
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

    nT = 295

    T_array = np.linspace(300, 15000, nT)

    # p, T, p_all
    nU = 2 + 8
    Z_table = np.zeros((n_p, nT, nU))

    for p_idx, p in enumerate(p_array):
        for T_idx, T in enumerate(T_array):
            if T_idx == 0:
                p_s0 = []
            else:
                p_s0 = model.p_all
            print("p:%.3e, T: %i"%(p, T))
            model.compute_pT(p, T, p_s0)

            # p
            Z_table[p_idx, T_idx, 0] = p
            # T
            Z_table[p_idx, T_idx, 1] = T
            # p_all
            Z_table[p_idx, T_idx, 2:] = model.p_all

    file = "RG8-results.dat"

    with open(file, 'wb') as f:
        np.save(f, Z_table)
    print("Data saved: %s"%file)