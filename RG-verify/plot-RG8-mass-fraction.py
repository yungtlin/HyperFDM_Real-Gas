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
    
    # plot setting
    rc = {"font.family" : "serif", 
          "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"] 

    colors = plt.cm.jet(np.linspace(0, 1, n_p))

    n_s = model.n_species

    for s_idx in range(n_s):
        fig, ax = plt.subplots(figsize=(5, 5)) 

        for p_idx in range(n_p):

            y_list = []

            T_array = data_RG8[p_idx, :, 1]

            for T_idx in range(n_T):
                p = data_RG8[p_idx, T_idx, 0]
                T = data_RG8[p_idx, T_idx, 1]
                p_all = data_RG8[p_idx, T_idx, 2:]

                model.update_p_all(T, p_all)
                y = np.log(model.c_all[s_idx])/np.log(10)

                y_list += [y]

            p_atm = p/101325
            ax.plot(T_array, y_list, color=colors[p_idx])
            
        lines = ax.get_lines()

        # Legend (color)
        label1 = ["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"]
        legend1 = plt.legend([lines[i] for i in range(n_p)], label1,
            title="Pressure (atm)", title_fontsize=9,
            fontsize=7, loc="upper center", ncol=len(label1),
            bbox_to_anchor=(0.5, 1.13), frameon=False)
        ax.add_artist(legend1)

        xlim = [0, 16000]
        ylim = [-20, 0]

        ax.set_xticks(np.linspace(xlim[0], xlim[1], 9))
        ax.set_xticks(np.linspace(xlim[0], xlim[1], 33), minor=True)
        ax.set_yticks(np.linspace(ylim[0], ylim[1], 6))
        ax.set_yticks(np.linspace(ylim[0], ylim[1], 21), minor=True)

        species_list = ["N_2", "O_2", "NO", "N", "O", "N^+", "O^+", "e^-"] 
        species_name = species_list[s_idx]

        plt.ylabel(r"$%s$ concentration ($\log_{10}(c_{%s})$)"%(species_name, species_name), fontsize=11)
        plt.xlabel(r"Temperature ($T$)", fontsize=11)
        ax.yaxis.set_label_coords(-0.09, 0.5)

        plt.ylim([-20, 1])
        plt.xlim([0, 15000])
        plt.grid(which="both")
        plt.savefig("RG8-c-%i-%s.png"%(s_idx, species_name))
        plt.savefig("RG8-c-%i-%s.pdf"%(s_idx, species_name))
        plt.clf()