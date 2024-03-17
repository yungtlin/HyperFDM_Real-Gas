###########
# Modules #
###########
import numpy as np
import sys
sys.path.append("../libs/real-gas/")
from Mixture import res_RG8_eta, solve_rhoe_RG8_lowtemp
from gas_models import get_model_RG8


if __name__ == "__main__":
    model = get_model_RG8()
    # N2: 0.79, O2:0.21
    model.init_composition([0.79, 0.21, 0, 0, 0, 0, 0, 0])

    p_atm_array = np.logspace(-4, 2, 7)

    p_array = p_atm_array*101325    
    n_p = p_atm_array.shape[0]

    nT = 295

    T_array = np.linspace(300, 15000, nT)

    T_idx = 10
    p_idx = 5
    T = T_array[T_idx]
    p = p_array[p_idx]


    model.compute_pT(p, T)
    eta_all = model.eta_all

    print(p ,T)

    rho = model.rho_mix
    e = model.e_mix


    model.compute_rhoe(rho, e, T0=1.02*T, eta0=eta_all*1.1)

    #err_T = np.abs(model.T - T)
    #err_p = np.abs(model.p_mix - p)
    #print("T err: %.3e, p err: %.3e"%(err_T, err_p))
    


