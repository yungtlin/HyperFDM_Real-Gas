###########
# Modules #
###########
import sys
import numpy as np 
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# self-defined libraries
sys.path.append("../real-gas/") #include RG8-model
from FDM import Zhong, FDM_interp
from Mesh_FDM_Full import Mesh
from gas_models import get_model_RG8
from Table_RG8 import Table_RG8_rhoe

##########
# Solver #
##########
# 2D Navier-Stokes shock-fitting solver
class Solver2D:
    ##################
    # Initialization #
    ##################
    def __init__(self, M_inf, N, gas_model="ideal"):
        # Gas related
        self.init_gas()
        self.set_RG8_table()
        self.set_gas_model(gas_model)

        # Given conditions
        self.set_freestream(M_inf)

        # Mesh
        self.N = N
        self.mesh = Mesh(N, self.r_c)
        self.mesh.init_H_polynomial(a_0=0.7, a_1=2, p=2)

        self.H = self.mesh.H
        self.Ht = self.mesh.Ht

        # Default stencil
        self.stencil = 5
        self.alpha = 0.25

        # Solution Space
        self.nU = 4
        self.U = np.zeros((self.nU, *N))
        self.V = np.zeros(self.U.shape)
        self.Eta_all = np.zeros((self.n_species, *N))
        self.set_init_freestream()

        # Parameter intialization
        self.total_iteration = 0
        self.T = 0
        self.set_flux_split("Zhong")


    def set_freestream(self, M_inf):
        # Given conditions (Zhong 1998)
        self.M_inf = M_inf
        self.T_inf = 300
        self.p_inf = 1e-3*101325
        self.T_w = 210.02
        self.Re = 2050
        self.r_c = 0.0061468

        self.model_RG8.compute_pT(self.p_inf, self.T_inf, is_print=False)

        gamma = self.gamma_ideal
        R_air = self.model_RG8.R_mix

        mu = viscosity_Suther(self.T_inf)

        self.U_inf = M_inf*np.sqrt(gamma*R_air*self.T_inf)

        # Re = u D / nu
        nu = self.U_inf*self.r_c/self.Re

        self.rho_inf = self.model_RG8.rho_mix
        self.E_inf = self.rho_inf*self.model_RG8.e_mix + 0.5*self.rho_inf*self.U_inf**2

    def set_init_freestream(self):
        gamma = self.gamma_ideal
        R_air = self.R_air_ideal

        c_inf = np.sqrt(gamma*R_air*self.T_inf)
        u_inf = self.M_inf*c_inf

        self.U[0] = self.rho_inf
        self.U[1] = self.rho_inf*u_inf
        self.U[2] = 0
        self.U[3] = self.p_inf/(gamma - 1) + 0.5*self.rho_inf*u_inf**2

        for s_idx in range(self.n_species):
            p_s = self.model_RG8.x0_all[s_idx]*self.p_inf
            self.Eta_all[s_idx] = p_s/(self.rho_inf*self.R_hat*self.T_inf)

    #######
    # Gas #
    #######
    def init_gas(self):
        # Gas properties
        self.model_RG8 = get_model_RG8()
        self.model_RG8.init_composition([0.79, 0.21, 0, 0, 0, 0, 0, 0])
        self.n_species = self.model_RG8.n_species

        self.R_hat = self.model_RG8.R_hat
        self.set_ideal_gas()

    def set_RG8_table(self):
        path = "../real-gas/RG8-rho-e-table.dat"
        self.interp_RG8 = Table_RG8_rhoe(path=path)
    
    def set_gas_model(self, gas_model):
        print("Using: %s gas model"%gas_model)
        if gas_model == "ideal":
            self.state_UtoV = self.state_UtoV_ideal
            self.state_VtoU = self.state_VtoU_ideal
        elif gas_model == "RG8":
            pass 
        else:
            raise ValueError("Gas model: %s NOT FOUND"%gas_model)

    def set_ideal_gas(self):
        # Gas properties
        self.gamma_ideal = 1.4
        self.Pr_ideal = 0.77
        self.R_air_ideal = 287.05

    def init_guess_RG8(self):
        print("Initializing gas composition with the interpolation...")
        rho_all = self.U[0]
        e_all = self.state_Utoe(self.U)

        u_interp = np.zeros(self.interp_RG8.n_z)
        for j in range(self.N[0]):
            for i in range(self.N[1]):
                rho = rho_all[j, i]
                e = e_all[j, i]

                u_interp = self.interp_RG8.get_data(rho, e)

                self.V[0, j, i] = u_interp[0] # p
                self.V[3, j, i] = u_interp[1] # T
                self.Eta_all[:, j, i] = u_interp[2:]
    
    def update_V(self):
        self.V = self.state_UtoV(self.U, self.V, self.Eta_all)

    #######
    # Run #
    #######
    def run_steady(self, max_iter, CFL, temporal="FE", tol_min=1e-4, print_step=1):
        # temporal method selection
        time_integration = self.time_selector(temporal)

        # Update
        self.update_FDM_stencil()

        # start iteration
        iteration = 0
        for iteration in range(max_iter):
            # time step estimation
            dt = self.time_get_dt(CFL)

            time_integration(dt)

            self.err, self.is_converged = self.run_is_converge_check(tol_min, dt)
            
            # update 
            self.T += dt
            self.total_iteration += 1

            if self.total_iteration % print_step == 0:
                self.run_steady_print_error() 

            if self.is_converged:
                self.run_steady_print_error() 
                print("\nSolution converged!")
                break
    
        # Update boundary condition for the result
        self.mesh.update_H(self.H, self.Ht)

        self.boundary_update_U(self.U, self.state_UtoV(self.U), self.mesh)
        print()

    def run_is_converge_check(self, tol_min, dt):
        v = self.U[2]/self.U[0]
        new_v = v

        # computing tolerence
        try:
            dev = self.old_v - new_v
        except AttributeError:
            tol = np.inf
        else:
            err_L1 = np.mean(np.abs(dev))
            tol = err_L1/(self.U_inf*dt)
        self.old_v = new_v

        # convergence criterions
        T_u = self.r_c/self.U_inf
        if tol < tol_min: # and self.T > 2.5*T_u:
            is_converged = True 
        else:
            is_converged = False
        return tol, is_converged

    def run_steady_print_error(self):
        print("Iteration: %i, T: %.5e, err: %.5e"\
            %(self.total_iteration, self.T, self.err), end="\r")
        sys.stdout.flush()


    ########
    # Time #
    ########
    def time_selector(self, method_name):
        if method_name == "FE":
            print("Set temporal: Forward Euler")
            time_method = self.time_update_FE
        elif method_name == "RK3":
            print("Set temporal: SSP-RK3")
            time_method = self.time_update_RK3
        else:
            raise ValueError("Method: %s Not Found"%method_name)

        return time_method

    # 1st-order Forward Euler
    def time_update_FE(self, dt):
        U0 = [self.U, self.H, self.Ht]

        K1 = self.flux_all(U0)
        U1 = self.time_update_step(U0, K1, dt)

        self.U, self.H, self.Ht = U1

    # SSP-RK3
    def time_update_RK3(self, dt):
        U0 = [self.U, self.H, self.Ht]

        # Stage 1
        L = self.flux_all(U0)
        U_tmp = self.time_update_SSP(U0, U0, L, dt, 1)

        # Stage 2
        L = self.flux_all(U_tmp)
        U_tmp = self.time_update_SSP(U0, U_tmp, L, dt, 1/4)

        # Stage 3
        L = self.flux_all(U_tmp)
        U_tmp = self.time_update_SSP(U0, U_tmp, L, dt, 2/3)

        [self.U, self.H, self.Ht] = U_tmp

    def time_update_step(self, U, Ut, dt):
        U1 = []
        for idx, u in enumerate(U):
            ut = Ut[idx]
            u1 = u + ut*dt
            U1 += [u1]

        return U1

    def time_update_SSP(self, Un, U_tmp, L, dt, a):
        U1 = []
        for idx, un in enumerate(Un):
            u_tmp = U_tmp[idx]
            l = L[idx]
            u1 = (1 - a)*un + a*(u_tmp + dt*l)
            U1 += [u1]

        return U1

    #####################################################
    # BUGS: Does not consider mesh moving speed in time #
    #####################################################
    # u_ref: local fastest speed
    # dx_ref: local 
    def time_get_dt(self, CFL):
        V = self.state_UtoV(self.U)
        u = V[1]
        v = V[2]
        T = V[3]

        c = np.sqrt(self.gamma_ideal*self.R_air_ideal*T)
        u_abs = np.sqrt(u**2 + v**2)
        u_ref = np.max(c + u_abs)

        dx_ref = self.mesh.get_dx_ref()

        dt_ref = CFL*dx_ref/u_ref

        dt = np.min(dt_ref)

        return dt

    ########
    # Flux #
    ########
    def flux_all(self, U_pack):
        # Unpack
        U, H, Ht = U_pack

        # Update Mesh
        self.mesh.update_H(H, Ht)

        # Cal primitive state
        V = self.state_UtoV(U)

        # Prior-Computation Boundary Update #
        self.boundary_update_U(U, V, self.mesh)

        # Flux evaluation
        E_inv, F_inv = self.flux_inviscid(U, V)

        # Flux transformation
        E = E_inv
        F = F_inv 
        E_hat, F_hat = self.flux_hat(U, E, F)

        # Flux splitting #
        E_hat_p, E_hat_m, F_hat_p, F_hat_m = self.flux_split(U, V, E_hat, F_hat)

        # FDM discretization #
        E1F1 = self.FDM_discretize(E_hat_p, E_hat_m, F_hat_p, F_hat_m)

        # Conservative form
        dUdt = self.flux_ALE(U, E_hat, F_hat, E1F1)

        # Shock Movment
        if self.is_shock_move:
            Htt = self.shock_move_acceleration(U, V, self.mesh, dUdt)
        else: 
            Htt = 0

        # Post-Boundary Update #
        self.boundary_update_dUdt(U, V, dUdt)


        return dUdt, Ht, Htt

    # Inviscid fluxes
    def flux_inviscid(self, U, V): 
        E = np.zeros(U.shape)
        F = np.zeros(U.shape)

        rho = U[0]
        rhou = U[1]
        rhov = U[2]
        e = U[3]

        p = V[0]
        u = V[1]
        v = V[2]
        T = V[3]

        # fluxes
        E[0] = rhou
        E[1] = rhou*u + p
        E[2] = rhou*v
        E[3] = (e + p)*u

        F[0] = rhov
        F[1] = rhov*u
        F[2] = rhov*v + p
        F[3] = (e + p)*v

        return E, F


    def flux_hat(self, U, E, F):
        E_hat = np.zeros(E.shape)
        F_hat = np.zeros(F.shape)

        x_xi = self.mesh.x_xi
        x_eta = self.mesh.x_eta
        y_xi = self.mesh.y_xi
        y_eta = self.mesh.y_eta

        geta_t = self.mesh.geta_t

        E_hat = y_eta*E - x_eta*F
        F_hat = -y_xi*E + x_xi*F + geta_t*U

        return E_hat, F_hat

    def set_flux_split(self, method):
        if method == "ABS":
            self.flux_split = self.flux_split_ABS
        elif method == "Zhong":
            self.flux_split = self.flux_split_Zhong

    def flux_split_Zhong(self, U, V, E_hat, F_hat, epsilon=0.5):
        rho = U[0]
        p = V[0]
        u = V[1]
        v = V[2]

        c = np.sqrt(self.gamma_ideal*p/rho)

        a = np.sqrt(self.mesh.y_eta**2 + self.mesh.x_eta**2) # g|\nabla \eta|
        ac = a*c
        lamb_E = np.sqrt((self.mesh.y_eta*u - self.mesh.x_eta*v)**2\
            + (epsilon*ac)**2) + ac

        b = np.sqrt(self.mesh.y_xi**2 + self.mesh.x_xi**2)
        bc = b*c
        lamb_F = np.sqrt((-self.mesh.y_xi*u + self.mesh.x_xi*v + self.mesh.geta_t)**2\
            + (epsilon*bc)**2) + bc

        # E_hat #
        E_hat_p = 0.5*(E_hat + lamb_E*U)
        E_hat_m = 0.5*(E_hat - lamb_E*U)

        # F_hat #
        F_hat_p = 0.5*(F_hat + lamb_F*U)
        F_hat_m = 0.5*(F_hat - lamb_F*U)

        return E_hat_p, E_hat_m, F_hat_p, F_hat_m

    def flux_ALE(self, U, E, F, E1F1):
        E_xi, F_eta = E1F1
        A = E_xi + F_eta
        dUdt = -self.mesh.J*A

        return dUdt

    #########
    # Shock #
    #########
    def set_is_shock_move(self, is_shock_move):
        if not is_shock_move:
            self.Ht = np.zeros(self.Ht.shape)
        self.is_shock_move = is_shock_move

    def shock_move_acceleration(self, U, V, mesh, dUdt):
        # surface properties
        n = mesh.surface_n
        v_n = mesh.surface_v_n
        n_t, c_shock = mesh.get_surface_shock_move()

        # behind the shock states
        U_s = U[:, -1]
        V_s = V[:, -1]
        E_s, F_s = self.flux_inviscid(U_s, V_s)
        Fs_nt = E_s*n_t[0] + F_s*n_t[1]
        dUsdt = dUdt[:, -1]

        # free stream states
        V_0 = np.array([[self.p_inf, self.U_inf, 0, self.T_inf]]).T
        U_0 = self.state_VtoU(V_0)
        E_0, F_0 = self.flux_inviscid(U_0, V_0)
        F0_nt = E_0*n_t[0] + F_0*n_t[1]

        lamb_N, I_N = self.shock_move_IN(U_s, V_s, v_n, n)

        #Bp_0 = self.shock_Jacobian_Bp(U_0, V_0, v_n, n)
        #Bp_s = self.shock_Jacobian_Bp(U_s, V_s, v_n, n)
        #IB = matmulbA3D(I_N, Bp_s)

        # dv/dt = -1/k1*(-k2 + k3)
        k1 = np.sum(I_N*(U_0 - U_s), axis=0)
        k2 = lamb_N*np.sum(I_N*dUsdt, axis=0)
        k3 = np.sum(I_N*(F0_nt - Fs_nt), axis=0)

        dvdt = -1/k1*(-k2 + k3)

        # (dvdt - c0)/c1
        Htt = (dvdt - c_shock[0])/c_shock[1]

        return Htt


    def shock_move_IN(self, U_s, V_s, v_n, n):
        rho = U_s[0]
        p = V_s[0]
        u = V_s[1]
        v = V_s[2]

        c = np.sqrt(self.gamma_ideal*p/rho)
        u_p = u*n[0] + v*n[1]
        alpha = (u**2 + v**2)/2
        beta = self.gamma_ideal - 1

        lamb_N = (u_p + v_n) - c

        I_N = np.zeros(U_s.shape)

        I_N[0] = alpha*beta + u_p*c
        I_N[1] = -u*beta - c*n[0]
        I_N[2] = -v*beta - c*n[1]
        I_N[3] = beta

        return lamb_N, I_N

    def shock_Jacobian_Bp(self, U, V, v_n, n):
        n_U, n_pt = U.shape
        A_shape = (n_U, n_U, n_pt)

        A = np.zeros(A_shape)
        B = np.zeros(A_shape)

        rho = U[0]
        u = V[1]
        v = V[2]
        E = U[3]/rho

        u_sq = u**2
        v_sq = v**2
        uv = u*v

        gamma = self.gamma_ideal

        # Jacobian matrix A (dE/dU)
        A[0, 1] = 1

        A[1, 0] = (gamma-3)/2*u_sq + (gamma-1)/2*v_sq
        A[1, 1] = (3-gamma)*u 
        A[1, 2] = (1-gamma)*v
        A[1, 3] = gamma-1

        A[2, 0] = -uv 
        A[2, 1] = v 
        A[2, 2] = u 

        A[3, 0] = -gamma*u*E + (gamma-1)*u*(u_sq + v_sq)
        A[3, 1] = gamma*E - (gamma-1)/2*(v_sq + 3*u_sq) 
        A[3, 2] = (1-gamma)*uv 
        A[3, 3] = gamma*u 

        # Jacobian matrix B (dF/dU)
        B[0, 2] = 1
    
        B[1, 0] = -uv 
        B[1, 1] = v 
        B[1, 2] = u 
    
        B[2, 0] = (gamma-3)/2*v_sq + (gamma-1)/2*u_sq 
        B[2, 1] = (1-gamma)*u 
        B[2, 2] = (3-gamma)*v
        B[2, 3] = gamma-1
    
        B[3, 0] = -gamma*v*E + (gamma-1)*v*(u_sq + v_sq)
        B[3, 1] = (1-gamma)*uv 
        B[3, 2] = gamma*E - (gamma-1)/2*(u_sq + 3*v_sq)
        B[3, 3] = gamma*v

        Bp = np.zeros((n_U, n_U, n.shape[1]))
        
        for j in range(n_U):
            for i in range(n_U):
                Bp[j, i] = A[j, i]*n[0] + B[j, i]*n[1]

                if i == j:
                    Bp[j, i] += v_n

        return Bp


    ############
    # Boundary #
    ############
    def set_boundary(self, top="constant", bot="inviscid", out="extra"):
        ## Top boundary (inflow) ##
        print("Set top boundary method to: %s"%top)
        if top == "constant":
            self.boundary_U_top = self.boundary_U_top_constant
            self.boundary_dUdt_top = self.boundary_dUdt_top_zero
            self.is_shock_move = False
        elif top == "SF":
            raise NotImplementedError("Shock-fitting Method is NOT implemented")
            #self.boundary_U_top = self.boundary_U_top_SF
            #self.boundary_dUdt_top = self.boundary_dUdt_pass
            #self.is_shock_move = True
        else:
            raise ValueError("Top BC: %s NOT FOUND"%top)

        ## Bottom boundary (wall) ##
        print("Set bottom boundary method to: %s"%bot)
        if bot == "inviscid":
            self.boundary_U_bot = self.boundary_U_wall_inviscid
            self.boundary_dUdt_bot = self.boundary_dUdt_wall_inviscid
        else:
            raise ValueError("Bottom BC: %s NOT FOUND"%bot)

        ## Outflow ##
        print("Set outflow boundary method to: %s"%out)
        if out == "dudt": 
            self.boundary_U_rght = self.boundary_U_out_dUdt
            self.boundary_dUdt_rght = self.boundary_dUdt_pass

            self.boundary_U_left = self.boundary_U_out_dUdt
            self.boundary_dUdt_left = self.boundary_dUdt_pass
        elif out == "extra":
            self.boundary_U_rght = self.boundary_U_rght_extra
            self.boundary_dUdt_rght = self.boundary_dUdt_pass

            self.boundary_U_left = self.boundary_U_left_extra
            self.boundary_dUdt_left = self.boundary_dUdt_pass
        else:
            raise ValueError("Right BC: %s NOT FOUND"%out)



    # Prior-Computation Boundary Update #
    def boundary_update_U(self, U, V, mesh):

        # Ouetflow (left)
        self.boundary_U_left(U, V)
        # Outflow (right)
        self.boundary_U_rght(U, V)

        # Wall (bottom)
        self.boundary_U_bot(U, V)

        # Inflow (top)
        self.boundary_U_top(U, V, mesh)

    # Flux Boundary Update #
    def boundary_update_dUdt(self, U, V, dUdt):
        # Outflow (right)
        self.boundary_dUdt_rght(dUdt)
        # Wall (bottom)
        self.boundary_dUdt_bot(U, V, dUdt)
        # Symmetric (left)
        self.boundary_dUdt_left(dUdt)
        # Inflow (top)
        self.boundary_dUdt_top(dUdt)

    ## Top ##
    # Constant condition (eta = -1)
    def boundary_U_top_constant(self, U, V, mesh):
        U[0, -1] = self.rho_inf
        U[1, -1] = self.rho_inf*self.U_inf
        U[2, -1] = 0
        U[3, -1] = self.E_inf

        V[:, -1] = self.state_UtoV(U[:, -1])

    def boundary_U_top_SF(self, U, V, mesh):
        n = mesh.surface_n

        gamma = self.gamma_ideal

        v_n = mesh.surface_v_n
        a_inf = np.sqrt(gamma*self.p_inf/self.rho_inf)

        u_n0 = self.U_inf*n[0]
        M_n0 = (u_n0 + v_n)/a_inf

        p_0 = self.p_inf
        rho_0 = self.rho_inf
        T_0 = self.T_inf

        p_s = p_0*(1 + 2*gamma/(gamma + 1)*(M_n0**2 - 1))
        rho_s = rho_0*((gamma + 1)*M_n0**2)/((gamma - 1)*M_n0**2 + 2)
        T_s = p_s*rho_0/(p_0*rho_s)*T_0
        u_ns = rho_0/rho_s*(u_n0 + v_n) - v_n

        u_s = self.U_inf - (u_n0 - u_ns)*n[0]
        v_s = - (u_n0 - u_ns)*n[1]

        V[0, -1, :] = p_s
        V[1, -1, :] = u_s
        V[2, -1, :] = v_s
        V[3, -1, :] = T_s

        U[:, -1] = self.state_VtoU(V[:, -1])

    def boundary_dUdt_top_zero(self, dUdt):
        dUdt[:, -1, :] = 0

    ## Bottom ##
    # u, v = 0 and T = T_w
    # dpdy = 0
    def boundary_U_wall_inviscid(self, U, V):
        mesh = self.mesh
        u = V[1, 0, :]
        v = V[2, 0, :]
        u_t, u_r = mesh.get_wall_velocity(u, v)

        # u_r = 0
        u = u - u_r*mesh.wall_n_n[0]
        v = v - u_r*mesh.wall_n_n[1]
        V[1, 0, :] = u
        V[2, 0, :] = v

        #dT/deta = 0
        # a0*T0 + an*Tn = 0
        # => T0 = b*Tn => b = -an/a0
        c_a = self.coeffs_eta[2][0]
        a0 = c_a[0]
        an = c_a[1:].reshape((-1, 1))
        stencil = c_a.shape[0]

        b = -an/a0
        Tn = V[3, 1:stencil, :]
        T_bot = np.sum(b*Tn, axis=0)
        V[3, 0, :] = T_bot

        # dp/deta = p*u_t**2*H/(R_air*T*r_c)
        # => dp/deta = p*c
        H = self.H
        r_c = self.r_c
        c = u_t**2*H/(self.R_air_ideal*T_bot*r_c)

        # a0*p0 + an*pn = p0*c
        # => p0 = (an*pn)/(c-a0)
        pn = V[0, 1:stencil, :]
        p_bot = np.sum(an*pn, axis=0)/(c - a0)
        V[0, 0, :] = p_bot
    
        # convert back
        U[:, 0] = self.state_VtoU(V[:, 0])


    def boundary_U_wall_extra(self, U, V):
        c_a = self.coeffs_extra_bot.reshape((-1, 1))

        stencil = c_a.shape[0] + 1

        p = V[0, 1:stencil, :]

        p_bot = np.sum(c_a*p, axis=0)

        V[0, 0, :] = p_bot

        # u, v = 0 and T = T_w
        V[1, 0, :] = 0
        V[2, 0, :] = 0
        V[3, 0, :] = self.T_w

        U[:, 0] = self.state_VtoU(V[:, 0])

    def boundary_dUdt_bot_zero(self, dUdt):
        dUdt[:, 0, :] = 0

    def boundary_dUdt_wall_inviscid(self, U, V, dUdt):
        mesh = self.mesh
        rho = U[0, 0, :]
        p = V[0, 0, :]
        u = V[1, 0, :]
        v = V[2, 0, :]
        u_t, u_r = mesh.get_wall_velocity(u, v)
        beta = mesh.beta
        r_c = self.r_c

        # 
        u_txi = mesh.get_FDM_xi(u_t, mesh.coeffs_xi)
        p_xi = mesh.get_FDM_xi(p, mesh.coeffs_xi)

        du_tdt = -(u_t/(beta*r_c)*u_txi + 1/(rho*beta*r_c)*p_xi)
        dudt = du_tdt*mesh.wall_n_t[0]
        dvdt = du_tdt*mesh.wall_n_t[1]

        dUdt[0, 0, :] = 0
        dUdt[1, 0, :] = rho*dudt
        dUdt[2, 0, :] = rho*dvdt
        dUdt[3, 0, :] = rho*(u*dudt + v*dvdt)

    ## Out flow ##

    ## Right ##
    # Uses solver results (works for supersonic outlet)
    def boundary_U_out_dUdt(self, U, V):
        pass

    # Do nothing
    def boundary_dUdt_pass(self, dUdt):
        pass

    # Extrapolation
    def boundary_U_left_extra(self, U, V):
        c_a = self.coeffs_extra_left.reshape((1, -1))

        stencil = c_a.shape[1] + 1

        for idx_v in range(self.nU):
            v = V[idx_v, :, 1:stencil]
            v_extra = np.sum(c_a*v, axis=1)
            V[idx_v, :, 0] = v_extra

        U[:, :, 0] = self.state_VtoU(V[:, :, 0])

    # Extrapolation
    def boundary_U_rght_extra(self, U, V):
        c_a = self.coeffs_extra_rght.reshape((1, -1))

        stencil = c_a.shape[1] + 1

        for idx_v in range(self.nU):
            v = V[idx_v, :, -stencil:-1]
            v_extra = np.sum(c_a*v, axis=1)
            V[idx_v, :, -1] = v_extra

        U[:, :, -1] = self.state_VtoU(V[:, :, -1])

    #######
    # FDM #
    #######
    def set_FDM_stencil(self, stencil, alpha):
        self.stencil = stencil
        self.alpha = alpha
        self.mesh.set_FDM_stencil(stencil)

    def update_FDM_stencil(self):
        stencil = self.stencil
        alpha = self.alpha

        print("FDM stencil: %i, alpha: %.2f"%(stencil, alpha))
        # xi direction (no flip)
        coeffs_p = self.FDM_Zhong_coeff(stencil, alpha, self.mesh.dxi)
        coeffs_m = self.FDM_Zhong_coeff(stencil, -alpha, self.mesh.dxi)
        coeffs_0 = self.FDM_Zhong_coeff(stencil, 0, self.mesh.dxi)
        self.coeffs_xi = np.array([coeffs_p, coeffs_m, coeffs_0])

        # eta direction (no flip)
        coeffs_p = self.FDM_Zhong_coeff(stencil, alpha, self.mesh.deta)
        coeffs_m = self.FDM_Zhong_coeff(stencil, -alpha, self.mesh.deta)
        coeffs_0 = self.FDM_Zhong_coeff(stencil, 0, self.mesh.deta)
        self.coeffs_eta = np.array([coeffs_p, coeffs_m, coeffs_0])

        # extrapolation 
        if stencil == 9:
            stencil_extra = stencil - 1
        else:
            stencil_extra = stencil

        loc_bot = self.mesh.Eta[1:(stencil_extra + 1), 0]
        self.coeffs_extra_bot = FDM_interp(0, loc_bot)

        loc_left = self.mesh.Xi[0, 1:stencil_extra] - self.mesh.Xi[0, 0]
        self.coeffs_extra_left = FDM_interp(0, loc_left)

        loc_rght = self.mesh.Xi[0, -stencil_extra:-1] - self.mesh.Xi[0, -1]
        self.coeffs_extra_rght = FDM_interp(0, loc_rght)

    # Uses (p-1) alpha=0 on boundary stencil 
    def FDM_Zhong_coeff(self, stencil, alpha_int, dxi):
        loc = np.arange(0, stencil)
        
        coeff_list = np.zeros((stencil, stencil))

        n_int = int((stencil-1)/2)

        if stencil == 3:
            n_fill = 1
        elif stencil > 3 and stencil < 9:
            n_fill = 2
        else: # 7th order used 5th order boundary
            n_fill = 3
        fill_0 = np.zeros(n_fill)

        for idx in range(stencil):
            # interior stencil
            if idx == n_int:
                N = stencil
                alpha = alpha_int
            
            # boundary stencils
            else:
                N = stencil - n_fill
                alpha = 0
            # left boundary
            if idx < n_int:
                N0 = N - idx - 1
            else:
                N0 = stencil - idx - 1

            M = 1
            M0 = 0

            form = (N, N0, M, M0)
            loc_idx = (loc - idx)*dxi

            c_a, c_b = Zhong(1, form, loc_idx, alpha=alpha)
            # right filling
            if idx < n_int:
                c_a = np.append(c_a, fill_0)
            # left filling
            elif idx > n_int:
                c_a = np.append(fill_0, c_a)
            coeff_list[idx] = c_a

        return coeff_list

    def FDM_discretize(self, E_hat_p, E_hat_m, F_hat_p, F_hat_m):
        # Xi direction
        coeff_xi_p = self.coeffs_xi[0]
        coeff_xi_m = self.coeffs_xi[1]

        E_p_xi = self.FDM_convolve_xi_all(E_hat_p, coeff_xi_p)
        E_m_xi = self.FDM_convolve_xi_all(E_hat_m, coeff_xi_m)

        E_xi = E_p_xi + E_m_xi

        # Eta direction 
        coeff_eta_p = self.coeffs_eta[0]
        coeff_eta_m = self.coeffs_eta[1]

        F_p_eta = self.FDM_convolve_eta_all(F_hat_p, coeff_eta_p)
        F_m_eta = self.FDM_convolve_eta_all(F_hat_m, coeff_eta_m)

        F_eta = F_p_eta + F_m_eta

        return E_xi, F_eta

    def FDM_convolve_xi_all(self, U, coeffs):
        nU = U.shape[0]
        U_xi = np.zeros(U.shape)
        for idx_u in range(nU):
            U_xi[idx_u] = self.FDM_convolve_xi(U[idx_u], coeffs)
        return U_xi

    def FDM_convolve_xi(self, U, coeffs):
        n_stencil, stencil_size = coeffs.shape
        n_c = round((stencil_size-1)/2)

        U_xi = np.zeros(U.shape)

        ## Convolution ##
        # Left Boundary
        u = U[:, :stencil_size]
        for idx in range(n_c):
            c_a = np.flip([coeffs[idx]])
            au = signal.convolve2d(c_a, u, mode="valid")
            U_xi[:, idx] = au.reshape(-1)

        # interior points
        c_a = np.flip([coeffs[n_c]])
        U_xi[:, n_c:-n_c]= signal.convolve2d(c_a, U, mode="valid")  

        u = U[:, -stencil_size:]
        for n_idx in range(n_c):
            idx = -n_c + n_idx
            c_a = np.flip([coeffs[idx]])
            au = signal.convolve2d(c_a, u, mode="valid")
            U_xi[:, idx] = au.reshape(-1)

        return U_xi

    def FDM_convolve_eta_all(self, U, coeffs):
        nU = U.shape[0]
        U_eta = np.zeros(U.shape)
        for idx_u in range(nU):
            U_eta[idx_u] = self.FDM_convolve_eta(U[idx_u], coeffs)
        return U_eta


    # Discretization (etaeta) #
    def FDM_convolve_eta(self, U, coeffs):
        n_stencil, stencil_size = coeffs.shape
        n_c = round((stencil_size-1)/2)

        U_eta = np.zeros(U.shape)

        ## Convolution ##
        # Left Boundary
        u = U[:stencil_size, :]
        for idx in range(n_c):
            c_a = np.flip([coeffs[idx]]).T
            au = signal.convolve2d(c_a, u, mode="valid")
            U_eta[idx, :] = au.reshape(-1)

        # Interior Points
        c_a = np.flip([coeffs[n_c]]).T

        aU = signal.convolve2d(c_a, U, mode="valid")
        U_eta[n_c:-n_c, :] = aU

        # Right Boundary
        u = U[-stencil_size:, :]
        for n_idx in range(n_c):
            idx = -n_c + n_idx
            c_a = np.flip([coeffs[idx]]).T
            au = signal.convolve2d(c_a, u, mode="valid")
            U_eta[idx, :] = (au).reshape(-1)

        return U_eta

    #########
    # State #
    #########
    # primitive states [p, u, v, T]
    def state_UtoV_ideal(self, U, V, Eta_all):
        gamma = self.gamma_ideal
        R_air = self.R_air_ideal

        V = np.zeros(U.shape)

        rho = U[0]
        u = U[1]/rho 
        v = U[2]/rho 
        E = U[3]/rho 
        cvRT = E - 1/2*(u**2 + v**2)
        T = (gamma-1)/R_air*cvRT
        p = rho*R_air*T

        V[0] = p 
        V[1] = u 
        V[2] = v 
        V[3] = T

        #check_negative_pressure(p)

        return V

    # primitive states [p, u, v, T]
    def state_VtoU_ideal(self, V, U, Eta_all):
        gamma = self.gamma_ideal
        R_air = self.R_air_ideal

        U = np.zeros(V.shape)

        p = V[0]
        u = V[1]
        v = V[2]
        T = V[3]

        rho = p/(R_air*T)
        e = p/(gamma - 1) + 0.5*rho*(u**2 + v**2)

        U[0] = rho
        U[1] = rho*u 
        U[2] = rho*v
        U[3] = e

        return U

    def state_Utoe(self, U):
        rho = U[0]
        u = U[1]/rho
        v = U[2]/rho
        e = U[3]/rho - 0.5*(u**2 + v**2)

        return e

    ##########
    # Remesh #
    ##########
    def remesh(self, N_new):
        # Mesh #
        mesh_new = Mesh(N_new, self.r_c)
        mesh_old = self.mesh

        xi_old = mesh_old.Xi[-1]
        xi_new = mesh_new.Xi[-1]

        H_old = self.H
        Ht_old = self.Ht

        H_new = interp1d(xi_new, xi_old, H_old)
        Ht_new = interp1d(xi_new, xi_old, Ht_old)

        # State #
        U_old = self.U
        U_new = self.remesh_state(mesh_new, mesh_old, U_old)

        # Assign #
        self.N = N_new
        self.mesh = mesh_new
        self.H = H_new
        self.Ht = Ht_new
        self.mesh.update_H(self.H, self.Ht)
        self.U = U_new


    def remesh_state(self, mesh_new, mesh_old, U_old):
        nU = U_old.shape[0]

        N_new = mesh_new.N
        U_new = np.zeros((nU, *N_new))

        xi_old = mesh_old.Xi[0, :]
        eta_old = mesh_old.Eta[:, 0]

        xi_new = mesh_new.Xi[0, :]
        eta_new = mesh_new.Eta[:, 0]

        for u_idx in range(nU):
            u_old = U_old[u_idx]
            f = interpolate.interp2d(xi_old, eta_old, u_old, kind='cubic')
            u_new = f(xi_new, eta_new)
            U_new[u_idx] = u_new

        return U_new

    #######
    # FIO #
    #######
    def save(self, folder, prefix):
        # get suffix
        order = self.stencil - 2
        ny, nx = self.N
        suffix = "_r%i_ny%i_nx%i.dat"%(order, ny, nx)
        file_path = folder + prefix + suffix
        print("Saving file: %s"%file_path)
        file = open(file_path, "wb") 

        # Header
        write_version = 1
        nX = [self.nU, self.N[0], self.N[1]]
        r = [self.r_c]
        writeline(file, np.int64, write_version)
        writeline(file, np.int64, self.n_species)
        writeline(file, np.int64, nX)
        writeline(file, np.float64, r)

        # grid 
        H = self.H
        Ht = self.Ht
        writeline(file, np.float64, H)
        writeline(file, np.float64, Ht)

        # solution
        for j in range(self.nU):
            for i in range(self.N[0]):
                writeline(file, np.float64, self.U[j, i])

        for j in range(self.n_species):
            for i in range(self.N[0]):
                writeline(file, np.float64, self.Eta_all[j, i])

    def load(self, path):
        print("Loading data: %s"%path)
        file = open(path, "rb")
        version = readline(file, np.int64, 1)[0]
        n_species = readline(file, np.int64, 1)[0]
        self.nU, Ny, Nx = readline(file, np.int64, 3)
        r_c = readline(file, np.float64, 1)
        self.N = [Ny, Nx]
        self.r_c = r_c[0]

        self.H = readline(file, np.float64, Nx)
        self.Ht = readline(file, np.float64, Nx)

     
        self.mesh = Mesh(self.N, self.r_c)
        self.mesh.update_H(self.H, self.Ht)

        self.U = np.zeros((self.nU, self.N[0], self.N[1]))
        for j in range(self.nU):
            for i in range(self.N[0]):
                self.U[j][i] = readline(file, np.float64, self.N[1])

        self.Eta_all = np.zeros((n_species, self.N[0], self.N[1]))
        for j in range(n_species):
            for i in range(self.N[0]):
                self.Eta_all[j][i] = readline(file, np.float64, self.N[1])

        file.close()

    ########
    # Plot #
    ########
    def plot_solution(self, U, level=11):
        r_c = self.mesh.r_c
        plt.contourf(self.mesh.X/r_c, self.mesh.Y/r_c, U, level, 
            cmap=cm.jet, extend="both")


#################
# Sub Functions #
#################
def viscosity_Suther(T):
    check_negative_temperature(T)

    mu_0 = 1.716e-5
    T_r = 273.15
    T_s = 110.4

    mu = mu_0*(T/T_r)**(3/2)*(T_r + T_s)/(T + T_s)

    return mu

def matmulbA3D(b, A):
    nb, nx = b.shape
    nbA, nA, nxA = A.shape

    if nb != nbA or nx != nxA:
        raise ValueError("matmulba3D Dimensions Are Not Matched")

    bA = np.zeros((nA, nx))
    for i in range(nA):
        bA[i] = np.sum((b*A[:, i]), axis=0)

    return bA

def interp1d(x, x_ref, y_ref):
    f = interpolate.interp1d(x_ref, y_ref, kind="cubic")
    y = f(x)

    return y

# write data to the bin file
def writeline(file, dtype, data):
    data_np = np.array(data, dtype=dtype)
    file.write(data_np.tobytes())

def readline(file, dtype, count):
    if dtype == np.float64:
        byte = 8
    elif dtype == np.int64:
        byte = 8 # Linux: 8 windows: 4

    byte_arr = file.read(byte*count)
    return np.frombuffer(byte_arr, dtype=dtype, count=count)

def check_negative_temperature(T):
    if np.min(T) < 0:
        raise RuntimeError("Temperature Below 0!!")

def check_negative_pressure(p):
    if np.min(p) < 0:
        raise RuntimeError("Pressure Below 0!!")

if __name__ == "__main__":
    N = [21, 21]
    M_inf = 10
    solver = Solver2D(M_inf, N)

    # 5pt, alpha = 0.25/0.5
    # 7pt, alpha = -6/-12
    # 9pt, alpha = 36/72
    stencil = 3
    alpha = -1

    solver.load("data/ideal_r1_ny21_nx21.dat")
    
    #solver.init_guess_RG8()

    solver.set_FDM_stencil(stencil, alpha)
    solver.set_boundary(out="dudt")

    #N = [81, 81]
    #solver.remesh(N)

    # Steady Run
    #max_iter = 3000
    #CFL = 0.8
    #solver.set_is_shock_move(False)
    #solver.run_steady(max_iter, CFL, tol_min=1e-2, temporal="RK3")

    # Shock moveboundary_dUdt_wall_inviscid
    #max_iter = 10000
    #CFL = 0.4
    #solver.set_is_shock_move(False)
    #solver.run_steady(max_iter, CFL, tol_min=1e-4, temporal="FE")

    #solver.save("data/", "ideal")
    
    #
    U = solver.U
    V = solver.V
    T = V[3]
    p = V[0]

    XX = T
    #XX = p/solver.p_inf

    #XX = V[0]

    plt.figure(figsize=(5.8, 7))
    plt.title(r"Euler - Ideal Gas ($M_\infty$: %.1f)"%(M_inf), fontsize=14)
    #solver.plot_solution(XX, level=np.linspace(4.25, 7.5, 14))
    solver.plot_solution(XX, level=11)

    solver.mesh.plot_mesh()
    cbar = plt.colorbar(format="%.0f")
    cbar.set_label("Temperature (K)")
    plt.ylabel(r"$y/R$", fontsize=14)
    plt.xlabel(r"$x/R$", fontsize=14)
    plt.xlim([-2, 0])
    plt.ylim([0, 3])
    #plt.xlim([-1.15, -0.9])
    #plt.ylim([0, 0.6])
    plt.grid()
    plt.show()
