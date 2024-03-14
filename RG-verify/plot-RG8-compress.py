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

    # load ref. data (Hansen-1958a)
    file_csv = "csv/compressibility.csv"
    data_csv = np.loadtxt(file_csv, skiprows=1, delimiter=",")
    
    # plot setting
    rc = {"font.family" : "serif", 
          "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"] 

    fig, ax = plt.subplots(figsize=(8, 5)) 
    colors = plt.cm.jet(np.linspace(0, 1, n_p))


    for p_idx in range(n_p):

        y_list = []

        T_array = data_RG8[p_idx, :, 1]

        for T_idx in range(n_T):
            p = data_RG8[p_idx, T_idx, 0]
            T = data_RG8[p_idx, T_idx, 1]
            p_all = data_RG8[p_idx, T_idx, 2:]

            model.update_p_all(T, p_all)
            y = model.Z

            y_list += [y]

        p_atm = p/101325
        ax.plot(T_array, y_list, color=colors[p_idx])
        
        T_csv = data_csv[:, 0]
        y_csv = data_csv[:, p_idx + 1]
        ax.plot(T_csv, y_csv, "d", color=colors[p_idx])

    lines = ax.get_lines()
    # Legend (color)
    label1 = ["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"]
    legend1 = plt.legend([lines[2*i] for i in range(n_p)], label1,
        title="Pressure (atm)", title_fontsize=9,
        fontsize=9, loc="upper center", ncol=len(label1),
        bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.add_artist(legend1)

    # Legend (style)
    label2 = ["Computed", "Hansen and Heims (1958a)"]
    legend2 = plt.legend([lines[i] for i in range(2)], label2,
        title="Source", title_fontsize=9, 
        loc="upper left", fontsize=9)
    ax.add_artist(legend2)


    xlim = [0, 16000]
    ylim = [0.4, 4.4]

    ax.set_xticks(np.linspace(xlim[0], xlim[1], 9))
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 33), minor=True)
    ax.set_yticks(np.linspace(1, 4, 4))
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 21), minor=True)

    plt.ylabel(r"Compressibility ($Z$)", fontsize=12)
    plt.xlabel(r"Temperature ($T$)", fontsize=12)

    plt.ylim(ylim)
    plt.xlim([0, 15000])
    plt.grid(which="both")
    plt.show()