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
    N = [41, 41] # (will be overwritten by load)
    M_inf = 10 # free stream Mach number

    # Load file
    load_folder = "data/"
    load_file = "RG8_M%i_r1_ny%i_nx%i.dat"%(M_inf, *N)
    load_path = load_folder + load_file
    
    # If RG8 table is not existed, create one
    try:    
        solver = Solver2D(M_inf, N)
    except FileNotFoundError:
        gen_rho_e_data() # 
        solver = Solver2D(M_inf, N)
    solver.load(load_path)

    # refine put here
    N_new = 2*(np.array(N) - 1) + 1
    solver.remesh(N_new)

    # FDM realted setting
    stencil = 3 # 1st-order
    alpha = -1  # fully upwind
    solver.set_FDM_stencil(stencil, alpha)
    solver.set_boundary(out="dudt")

    # Starting RG8 (equilibrium) simulation
    solver.set_gas_model("RG8")    
    solver.init_guess_RG8()

    CFL = 0.4
    iteration = 5
    solver.run_steady(iteration, CFL, tol_min=1e-4, temporal="FE")
    solver.save(load_folder, "RG8_M%i"%M_inf)
