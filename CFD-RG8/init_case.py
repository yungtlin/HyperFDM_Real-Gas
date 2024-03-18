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
    # create data saving folder
    folder = "data"
    os.system("mkdir -p %s"%folder)

    # Case
    N = [21, 21] # grid points
    M_inf = 10 # free stream Mach number
    
    # If RG8 table is not existed, create one
    try:    
        solver = Solver2D(M_inf, N)
    except FileNotFoundError:
        gen_rho_e_data() # 
        solver = Solver2D(M_inf, N)
    
    # Mesh domain size
    solver.mesh.init_H_polynomial(a_0=0.7, a_1=2, p=2)
    solver.H = solver.mesh.H
    solver.Ht = solver.mesh.Ht

    # FDM realted setting
    stencil = 3 # 1st-order
    alpha = -1  # fully upwind
    solver.set_FDM_stencil(stencil, alpha)
    solver.set_boundary(out="dudt")

    # Ideal gas simulation 
    solver.set_gas_model("ideal")
    max_iter = 30000 # 
    CFL = 0.4
    solver.run_steady(max_iter, CFL, tol_min=1e-4, temporal="FE")
    solver.save("%s/"%folder, "ideal_M%i"%M_inf)

    # Starting RG8 (equilibrium) simulation
    solver.set_gas_model("RG8")    
    solver.init_guess_RG8() # guess equilibrium state with RG8 table
    max_iter = 10 # short run
    CFL = 0.4
    solver.run_steady(max_iter, CFL, tol_min=1e-4, temporal="FE")
    solver.save("%s/"%folder, "RG8_M%i"%M_inf)