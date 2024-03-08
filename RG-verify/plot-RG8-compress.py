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

    p_atm = 100

    p = p_atm*101325
    T = 1000

    nT = 59
    T_array = np.linspace(500, 15000, nT)
    Z_array = np.zeros(nT)

    for idx, T in enumerate(T_array):
        if idx == 0:
            p_s0 = []
        else:
            p_s0 = model.p_all
        print("T: %i"%T)
        model.compute_pT(p, T, p_s0)

        Z_array[idx] = model.Z

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(T_array, Z_array, "-b")

    xlim = [0, 15000]
    ylim = [0, 4]

    ax.set_xticks(np.linspace(xlim[0], xlim[1], 16))
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 31), minor=True)
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 21), minor=True)

    plt.ylim(ylim)
    plt.xlim([0, 15000])
    plt.grid(which="both")
    plt.show()

