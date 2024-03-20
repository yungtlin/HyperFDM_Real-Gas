###########
# Modules #
###########
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys, os
sys.path.append("../libs/FDM")
sys.path.append("../libs/real-gas")
from FDM_EU import Solver2D
from gen_rho_e_table import gen_rho_e_data

if __name__ == "__main__":
    # Case
    model = "ideal"
    M_inf = 30 # free stream Mach number
    N = [41, 41] # (will be overwritten by load)

    # Load file
    load_folder = "data/"
    load_file = "%s_M%i_r1_ny%i_nx%i.dat"%(model, M_inf, *N)
    load_path = load_folder + load_file

    # If RG8 table is not existed, create one
    try:    
        solver = Solver2D(M_inf, N)
    except FileNotFoundError:
        gen_rho_e_data() # 
        solver = Solver2D(M_inf, N)
    solver.load(load_path)

    # Starting RG8 (equilibrium) simulation
    solver.set_gas_model(model)
    #solver.init_guess_RG8()
    solver.update_V()
    XX = solver.V[3]

    plt.figure(figsize=(5.8, 7))
    level = np.linspace(10000, 55000, 46)
    ticks = np.linspace(10000, 55000, 10) 

    # plot setting
    rc = {"font.family" : "serif", 
          "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"] 

    title = "Ideal Gas" if model == "ideal" else "Real Gas"

    #level = np.linspace(0, 15000, 101)
    plt.title(r"%s ($M_\infty$: %.1f)"%(title, M_inf), fontsize=14)
    solver.plot_solution(XX, level=level)
    print(np.max(XX))
    #solver.mesh.plot_mesh()

    cbar = plt.colorbar(format="%.0f", ticks=ticks)
    #cbar = plt.colorbar(format="%.0f")
    cbar.set_label("Temperature (K)")
    plt.ylabel(r"$y/R$", fontsize=14)
    plt.xlabel(r"$x/R$", fontsize=14)
    plt.xlim([-2, 0])
    plt.ylim([0, 3])
    plt.grid()
    file = "%s_0_M%i_n%i"%(model, int(M_inf), N[0])
    plt.savefig(file + ".pdf")
    plt.savefig(file + ".png")