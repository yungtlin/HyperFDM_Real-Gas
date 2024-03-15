###########
# Modules #
###########
import math
import numpy as np
from scipy.interpolate import LinearNDInterpolator


##########
# Table #
##########
# 2D Navier-Stokes shock-fitting solver
class Table_RG8_rhoe:
    def __init__(self, path="RG8-rho-e-table.dat"):
        print("Loading table: %s"%path)

        try:
          self.data_RG8 = np.load(path, mmap_mode="r")
        except FileNotFoundError:
            raise FileNotFoundError("File: %s not found."%path
                + " Please use gen_rho_e_table.py to generate .dat file")

        self.create_rhoe_table()

    def create_rhoe_table(self):
        n_U, n_p, n_T  = self.data_RG8.shape

        rho = self.data_RG8[0]
        e = self.data_RG8[1]

        X = np.log(rho)
        Y = np.log(e)

        x = X.reshape(-1)
        y = Y.reshape(-1)


        self.interp_funcs = []
        self.n_z = n_U - 2
        for u_idx in range(2, n_U):
            U = self.data_RG8[u_idx]
            if u_idx != 3: # T
                U = np.log(U)
            u = U.reshape(-1)


            self.interp_funcs += [LinearNDInterpolator(list(zip(x, y)), u)]


    def get_data(self, rho, e):
        x = np.log(rho)
        y = np.log(e)

        u_array = np.zeros(self.n_z)

        for z_idx in range(self.n_z):
            u = self.interp_funcs[z_idx](x, y)
            if math.isnan(u):
                print("OOB: rho: %.3e, e: %.e"%(rho, e))
                raise ValueError("Table interpolation is Out Of Bound")

            if z_idx != 1: # T
                u = np.exp(u)

            u_array[z_idx] = u

        return u_array


if __name__ == "__main__":    
    table = Table_RG8_rhoe()

    rho_0 = 0.0225
    e_0 = 1.4e7

    u_array = table.get_data(rho_0, e_0)    

    print(u_array)