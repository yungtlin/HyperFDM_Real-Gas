###########
# Modules #
###########
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import interpolate

import sys, os
sys.path.append("../libs/FDM")
sys.path.append("../libs/real-gas")
from FDM_EU import Solver2D
from gen_rho_e_table import gen_rho_e_data

def interp_a12(M):
    M_array = np.array([3, 5, 7, 10, 15, 20, 30])
    a1_array = np.array([1, 0.8, 0.7, 0.7, 0.5, 0.5, 0.5])
    a2_array = np.array([3, 2, 2, 2, 1.6, 1.6, 1.6])

    f1 = interpolate.interp1d(M_array, a1_array)
    f2 = interpolate.interp1d(M_array, a2_array)

    a1 = f1(M)
    a2 = f2(M)
    return a1, a2

if __name__ == "__main__":
    N = [41, 41] # grid points
    
    M_array = np.array([3, 5, 7, 10, 15, 20, 30])

    U_list = []
    T_ideal_list = []
    T_RG8_list = []

    for M_inf in M_array:
        print("M: %.1f"%M_inf)
        # Case
        a1, a2 = interp_a12(M_inf)

        # If RG8 table is not existed, create one
        solver = Solver2D(M_inf, N)
        
        # Mesh domain size
        solver.mesh.init_H_polynomial(a_0=a1, a_1=a2, p=2)
        solver.H = solver.mesh.H
        solver.Ht = solver.mesh.Ht

        # FDM realted setting
        stencil = 3 # 1st-order
        alpha = -1  # fully upwind
        solver.set_FDM_stencil(stencil, alpha)
        solver.set_boundary(out="dudt")

        # Ideal gas simulation 
        solver.set_gas_model("ideal")
        solver.load("data/ideal_M%i_r1_ny41_nx41.dat"%M_inf)
        #max_iter = 30000 # 
        #CFL = 0.4
        #solver.run_steady(max_iter, CFL, tol_min=1e-0, temporal="FE")
        T_ideal = np.max(solver.V[3])

        # Starting RG8 (equilibrium) simulation
        solver.set_gas_model("RG8")    
        solver.load("data/RG8_M%i_r1_ny41_nx41.dat"%M_inf)
        #solver.init_guess_RG8() # guess equilibrium state with RG8 table
        #max_iter = 1 # short run
        #CFL = 0.4
        #solver.run_steady(max_iter, CFL, tol_min=1e-4, temporal="FE")
        T_RG8 = np.max(solver.V[3])

        U_list += [solver.U_inf]
        T_ideal_list += [T_ideal]
        T_RG8_list += [T_RG8]
        print()

    # load ref. data (Hansen-1958a)
    file_csv = "csv/Mach-temp.csv"
    data_csv = np.loadtxt(file_csv, skiprows=1, delimiter=",").T

    U_list = np.array(U_list)/1000

    T_array = data_csv[0]*1000
    U_ideal = data_csv[1]
    U_RG8 = data_csv[2]

    # plot setting
    rc = {"font.family" : "serif", 
          "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"] 

    fig, ax = plt.subplots(figsize=(6, 4)) 
    ax.plot(U_ideal, T_array, "-b")
    ax.plot(U_RG8, T_array, "-r")

    ax.plot(U_list, T_ideal_list, "sb")
    ax.plot(U_list, T_RG8_list, "sr")

    lines = ax.get_lines()
    # Legend (color)
    label1 = ["Ideal", "Equilibrium"]
    legend1 = plt.legend([lines[0], lines[1]], label1,
        title="Gas Model", title_fontsize=10,
        fontsize=9, loc="upper center", ncol=len(label1),
        bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax.add_artist(legend1)

    # Legend (style)
    label2 = ["Anderson (1984)", "Computed"]
    legend2 = plt.legend([lines[0], lines[2]], label2,
        title="Source", title_fontsize=9, 
        loc="lower right", fontsize=9)
    ax.add_artist(legend2)

    plt.xlabel(r"Velocity ($km/s$)", fontsize=10, y=1.2)
    plt.ylabel(r"Temperature ($K$)", fontsize=10, x=1.2)

    ax.yaxis.set_label_coords(-0.11, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.08)

    plt.grid()
    plt.ylim([0, 15000])
    plt.xlim([0, 15])

    file = "Mach-temp"
    plt.savefig(file + ".png")
    plt.savefig(file + ".pdf")
