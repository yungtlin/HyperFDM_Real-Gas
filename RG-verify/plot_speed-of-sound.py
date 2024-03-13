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

    p_idx = 2
    T_idx = 150

    data_info = data_RG8[p_idx, T_idx]

    p = data_info[0]
    T = data_info[1]
    p_all = data_info[2:]
    model.update_p_all(T, p_all)


    model.compute_a()

