###########
# Modules #
###########
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../libs/real-gas/")
from Species import Monatomic, Diatomic
from Mixture import Reaction, Mixture, res_RG8, res_RG8_prime 
from gas_models import get_model_RG8


if __name__ == "__main__":
    model = get_model_RG8()
    n_s = len(model.species_list)

    #model.print_info()

    p_mix = 101325
    T = 5000

    dp_array = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    #dp_array = np.array([1e-7])
    err_list = []

    for dp in dp_array:
        model.init_composition([0.79, 0.21, 0.02, 0.05, 0.07, 0.04, 0.01, 1e-3])

        p_0 = p_mix*model.x0_all

        p_0 = np.array([7.92707301e+01, 2.03054798e+01, 8.11870550e-01, 2.73816643e-06,
            9.36916799e-01, 2.79899932e-22, 2.34349282e-14, 2.34349285e-14])


        K_p_all = model.compute_K_p(T)
        ratio_NO = model.ratio_NO
        
        res_0 = np.array(res_RG8(p_0, p_mix, K_p_all, ratio_NO))
        res_prime_0 = res_RG8_prime(p_0, p_mix, K_p_all, ratio_NO)

        res_fdm = np.zeros((n_s, n_s))
        for idx in range(n_s):
            p = np.array(p_0)

            p[idx] += dp

            res_dp = np.array(res_RG8(p, p_mix, K_p_all, ratio_NO))


            res_fdm[:, idx] = (res_dp - res_0)/dp

        dev = res_fdm - res_prime_0

        error = np.max(np.abs(dev))

        err_list += [error]


    plt.loglog(dp_array, err_list, "-ob")
    plt.grid()
    plt.ylabel(r"$L^{\infty}$ error", fontsize=14)
    plt.xlabel(r"$\Delta p$", fontsize=14)
    plt.show()
        