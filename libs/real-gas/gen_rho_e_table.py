###########
# Modules #
###########
import numpy as np
import sys
from gas_models import get_model_RG8

def gen_rho_e_data(path="RG8-rho-e-table.dat"):
    model = get_model_RG8()
    # N2: 0.79, O2:0.21
    model.init_composition([0.79, 0.21, 0, 0, 0, 0, 0, 0])

    p_atm_array = np.logspace(-4, 2, 13)

    p_array = p_atm_array*101325
    n_p = p_atm_array.shape[0]

    T_array = np.append([298], np.linspace(300, 15000, 295))
    n_T = T_array.shape[0]

    # rho, e, p, T, eta_all
    nU = 4 + 8
    Z_table = np.zeros((nU, n_p, n_T))

    for p_idx, p in enumerate(p_array):
        for T_idx, T in enumerate(T_array):
            if T_idx == 0:
                p_s0 = []
            else:
                p_s0 = model.p_all
            print("p:%.3e, T: %i"%(p, T))
            model.compute_pT(p, T, p_s0)

            # rho
            Z_table[0, p_idx, T_idx] = model.rho_mix
            # e
            Z_table[1, p_idx, T_idx] = model.e_mix
            # p
            Z_table[2, p_idx, T_idx] = p
            # T
            Z_table[3, p_idx, T_idx] = T
            # p_all
            Z_table[4:, p_idx, T_idx] = model.eta_all

    with open(path, 'wb') as f:
        np.save(f, Z_table)
    print("Data saved: %s"%path)


if __name__ == "__main__":
    # load simulation results
    file = "RG8-rho-e-table.dat"
    try:
        data_RG8 = np.load(file, mmap_mode="r")
    except FileNotFoundError:
        gen_rho_e_data() # 
        data_RG8 = np.load(file, mmap_mode="r")
    n_U, n_p, n_T = data_RG8.shape

    rho = data_RG8[0]
    e = data_RG8[1]
    p = data_RG8[2]
    T = data_RG8[3]
    eta_all = data_RG8[4:]
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    rc = {"font.family" : "serif", 
          "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"] 


    U = np.log(eta_all[7])
    X = np.log(rho)
    Y = np.log(e)
    U = T

    from scipy.interpolate import LinearNDInterpolator
    x = X.reshape(-1)
    y = Y.reshape(-1)
    u = U.reshape(-1)
    interp = LinearNDInterpolator(list(zip(x, y)), u)

    fig, ax = plt.subplots(figsize=(7, 5)) 

    rho_0 = 0.0225
    e_0 = 1.4e7
    x_0 = np.log(rho_0)
    y_0 = np.log(e_0)
    u0 = interp(x_0, y_0)


    #plt.plot(x_0, y_0, "xk")
    #print(u0)

    level = np.linspace(1000, 15000, 50) 
    plt.plot(x, y, "sr", markersize=2, label="source data")
    plt.legend(fontsize=10)

    plt.contourf(X, Y, U, level, 
            cmap=cm.jet, extend="both")

    ticks = np.linspace(1000, 15000, 8) 
    cbar = plt.colorbar(ticks=ticks)
    cbar.set_label("Temperature (K)")


    plt.ylabel(r"Internal energy ($\ln(e)$)", fontsize=12)
    plt.xlabel(r"Density ($\ln(\rho)$)", fontsize=12)
    plt.ylim([12, 19])
    plt.savefig("interp_table.png")
    plt.savefig("interp_table.pdf")
