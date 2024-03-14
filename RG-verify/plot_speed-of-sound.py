###########
# Modules #
###########
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../libs/real-gas/")
from gas_models import get_model_RG8


if __name__ == "__main__":
    # initialize gas model
    model = get_model_RG8()
    # N2: 0.79, O2:0.21
    model.init_composition([0.79, 0.21, 0, 0, 0, 0, 0, 0])

    # load simulation results
    file = "RG8-results.dat"
    data_RG8 = np.load(file, mmap_mode="r")
    n_p, n_T, n_U = data_RG8.shape

    c_table = []

    for p_idx in range(n_p):
        c_list = []

        for T_idx in range(1, n_T):
            data_info = data_RG8[p_idx, T_idx]

            p = data_info[0]
            T = data_info[1]
            p_all = data_info[2:]
            model.update_p_all(T, p_all)

            rho = model.rho_mix

            a = model.compute_a()
            c = a**2*rho/p

            c_list += [c]
        c_table += [c_list]


    c_table = np.array(c_table)


    # plot setting
    rc = {"font.family" : "serif", 
          "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"] 

    fig, ax = plt.subplots(figsize=(8, 5)) 
    colors = plt.cm.jet(np.linspace(0, 1, n_p))


    T_array = data_RG8[0, :, 1][1:]
    for p_idx in range(n_p):
        c_array = c_table[p_idx]
        ax.plot(T_array, c_array, "-", color=colors[p_idx])

    plt.show()


