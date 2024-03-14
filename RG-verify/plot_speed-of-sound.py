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
    file_csv = "csv/speed-of-sound-b.csv"
    data_csv = np.loadtxt(file_csv, skiprows=1, delimiter=",")

    c_table = []

    T_skip = 24

    for p_idx in range(n_p):
        c_list = []

        for T_idx in range(T_skip, n_T):
            data_info = data_RG8[p_idx, T_idx]

            p = data_info[0]
            T = data_info[1]
            p_all = data_info[2:]
            model.update_p_all(T, p_all)

            rho = model.rho_mix

            a = model.compute_a()
            c = a**2*rho/p

            print("p: %.3e, T: %.3e, c:%.3e"%(p, T, c))

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

    T_array = data_RG8[0, :, 1][T_skip:]
    for p_idx in range(n_p):
        c_array = c_table[p_idx]
        ax.plot(T_array, c_array, "-", color=colors[p_idx])

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
    label2 = ["Computed", "Hansen and Heims (1958b)"]
    legend2 = plt.legend([lines[i] for i in range(2)], label2,
        title="Source", title_fontsize=9, 
        loc="upper left", fontsize=9)
    ax.add_artist(legend2)

    xlim = [0, 16000]
    ylim = [0.9, 1.7]

    ax.set_xticks(np.linspace(xlim[0], xlim[1], 9))
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 33), minor=True)
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 9))
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 17), minor=True)

    plt.ylabel(r"Speed of Sound ($\frac{a^2 \rho}{p}$)", fontsize=12)
    plt.xlabel(r"Temperature ($T$)", fontsize=12)

    plt.ylim(ylim)
    plt.xlim([0, 15000])
    plt.grid(which="both")

    plt.show()


