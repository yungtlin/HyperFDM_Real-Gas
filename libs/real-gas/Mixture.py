###########
# Modules #
###########
import numpy as np
#from scipy.optimize import fsolve, newton, root

############
# Reaction #
############
class Reaction:
    def __init__(self, reactants, products):
        self.load_reactants(reactants)
        self.load_products(products)
        self.formula = self.get_formula()
        self.D = self.get_D()

        self.T = 0

    def load_reactants(self, reactants):
        self.reactants = {}

        for parcel in reactants:
            nu, species  = parcel
            key = species.name
            if key in self.reactants.keys():
                raise ValueError("Speices %s is already existed in the reactants"%key)
            else:
                self.reactants[key] = (nu, species)

    def load_products(self, products):
        self.products = {}

        for parcel in products:
            nu, species  = parcel
            key = species.name
            if key in self.products.keys():
                raise ValueError("Speices %s is already existed in the products"%key)
            else:
                self.products[key] = (nu, species)

    def get_formula(self):
        formula = ""
        # reactants
        for count, key in enumerate(self.reactants.keys()):
            if count > 0:
                formula += " + "
            nu, species = self.reactants[key]
            formula += "%i*%s"%(nu, species.name)

        formula += " = "

        # products
        for count, key in enumerate(self.products.keys()):
            if count > 0:
                formula += " + "
            nu, species = self.products[key]
            formula += "%i*%s"%(nu, species.name)

        return formula

    def get_D(self):
        D = 0
        sign_list = [1, -1] 
        for idx, species_dict in enumerate([self.reactants, self.products]):
            sign = sign_list[idx]
            for key in species_dict.keys():
                nu, species = species_dict[key]
                epsilon = species.e_0*species.m
                D += sign*nu*epsilon
        return D

    # set 
    def set_T(self, T):
        if self.T != T:
            # set all
            for species_dict in [self.reactants, self.products]:
                for key in species_dict.keys():
                    nu, species = species_dict[key]
                    species.set_T(T)
            self.T = T

    ## TBD ##
    def check_atom_conserve(self):
        pass

    #############
    # Computing #
    #############
    def compute_K_p(self, T):
        self.set_T(T)

        # nu_all = nu_reac - nu_prod
        # fq_all = fQ_int^p/fQ_int^r
        nu_all = 0
        fQ_all = 1

        sign_list = [1, -1] 
        for idx, species_dict in enumerate([self.reactants, self.products]):
            sign = sign_list[idx]
            for key in species_dict.keys():
                nu, species = species_dict[key]

                fQ_all *= np.power(species.fQ_int, -sign*nu)    
                nu_all += sign*nu
        k = species.k
        K_p = np.power(1/(k*T), nu_all)*fQ_all*np.exp(self.D/(k*T))
        
        self.K_p = K_p


###########
# Mixture #
###########
class Mixture:
    def __init__(self, species_list, reaction_list):
        self.set_species(species_list)
        self.set_reaction(reaction_list)

        self.init_global()

    def set_species(self, species_list):
        self.species_list = species_list
        self.n_species = len(species_list)

    def set_reaction(self, reaction_list):
        self.reaction_list = reaction_list
        self.n_reaction = len(reaction_list)

    def init_global(self):
        self.R_hat = self.species_list[0].R_hat

    def set_T(self, T):
        self.T = T
        for species in self.species_list:
            species.set_T(T)

    # in terms of mole ratio 
    def init_composition(self, weights):
        assert len(self.species_list) == len(weights)

        w_array = np.array(weights)
        self.x0_all = w_array/np.sum(w_array)

        xM_all = np.zeros(self.x0_all.shape)
        for idx, species in enumerate(self.species_list):
            xM_all[idx] = self.x0_all[idx]*species.M_hat
        self.M_bar_mix0 = np.sum(xM_all) # weight per mole


        # O/N
        self.ratio_NO = self.get_ratio(self.species_list, self.x0_all)

    def get_ratio(self, species_list, c):
        return c[0]/c[1]

    ###########
    # Compute #
    ###########
    def compute_p_all(self, p_mix, eta_all):
        M_hat_mix = 1/np.sum(eta_all)
        p_all = p_mix*M_hat_mix*eta_all
        return p_all

    def compute_pT(self, p_mix, T, p_s0=[], is_print=True):

        self.set_T(T)
        if len(p_s0) == 0:
            p_all = p_mix*self.x0_all + 1e-20 # prevent zero division
        else: 
            assert len(p_s0) == self.n_species
            p_all = p_s0   

        K_p_all = self.compute_K_p(T)
        ratio_NO = self.ratio_NO

        self.p_all = solve_pT_RG8_Newton(p_all, p_mix, K_p_all, ratio_NO,
            is_print=is_print)
        self.compute_all()

    def compute_rhoe(self, rho, e, T0=0, eta0=[], is_print=True):
        T = T0 if T0 != 0 else e/(3*self.R_mix) 
        
        R_air_ideal = 288.27
        if len(eta0) == 0:
            eta_all = self.x0_all*R_air_ideal/self.R_hat + 1e-20
        else:
            assert len(eta0) == self.n_species
            eta_all = eta0       

        if T > 1800:
            self.eta_all, T = solve_rhoe_RG8_Newton(eta_all, T, rho, e, self,
                is_print=is_print)
        else:
            self.eta_all, T = solve_rhoe_RG8_lowtemp(eta_all, T, rho, e, self,
                is_print=is_print)

        self.rho_mix = rho
        self.set_T(T)
        self.compute_all_eta(self.eta_all)

        err = np.abs((self.e_mix - e)/e)
        if err > 1e-5:
            raise ValueError("rho e not converged!")

    def compute_rhoT1(self, rho, T):
        self.rho_mix = rho
        self.set_T(T)

        R_hat = self.species_list[0].R_hat
        K_p_all = self.compute_K_p(T)

        eta_all = solve_rhoT1(self, K_p_all[0])

        self.compute_all_eta(eta_all)
        self.rho_mix = rho

    def compute_K_p(self, T):
        self.set_T(T)
        K_p_all = np.zeros(self.n_reaction)
        for idx, reaction in enumerate(self.reaction_list):
            reaction.compute_K_p(T)
            K_p_all[idx] = reaction.K_p

        return K_p_all

    def compute_a(self):
        Jacobian = get_RG8_a_jacobian(self)


        n_s = self.n_species
        dp = 1
        dU = np.zeros((n_s + 2, 1))
        dU[-2] = dp

        J_inv = np.linalg.inv(Jacobian)

        x = np.matmul(J_inv, dU)

        drho = x[-2, 0]

        # imaginary values
        a = np.sqrt(dp/drho)
        
        return a

    # required p_all computed
    def compute_all(self):
        self.p_mix = np.sum(self.p_all)
        self.x_all = self.p_all/self.p_mix

        xM_all = np.zeros(self.n_species)
        e_all = np.zeros(self.n_species)
        cv_all = np.zeros(self.n_species)
        s_all = np.zeros(self.n_species)

        for idx, species in enumerate(self.species_list):
            xM_all[idx] = self.x_all[idx]*species.M_hat
            e_all[idx] = species.e_total
            cv_all[idx] = species.cv_total

            species.compute_entropy(self.p_all[idx])
            s_all[idx] = species.s_total 

        self.M_hat_mix = np.sum(xM_all)
        self.c_all = xM_all/self.M_hat_mix
        self.R_mix = species.R_hat/self.M_hat_mix

        self.e_mix = np.sum(self.c_all*e_all)
        self.cv_mix = np.sum(self.c_all*cv_all)
        self.s_mix = np.sum(self.c_all*s_all)

        self.h_mix = self.e_mix + self.R_mix*self.T

        self.rho_mix = self.p_mix/(self.R_mix*self.T)

        self.eta_all = self.p_all/(self.rho_mix*self.R_hat*self.T)

        M_bar_mix = 1/np.sum(self.eta_all)
        self.Z = self.M_bar_mix0/M_bar_mix


    def compute_all_eta(self, eta_all):
        R_hat = self.species_list[0].R_hat
        self.p_all = self.rho_mix*R_hat*self.T*eta_all

        self.compute_all()

    ##########
    # Update #
    ##########
    def update_p_all(self, T, p_all):
        assert len(p_all) == self.n_species
        self.set_T(T)
        self.p_all = p_all
        self.compute_all()

    #########
    # Print #
    #########
    def print_info(self):
        self.print_species()
        self.print_reactions()

    def print_species(self):
        str_species = ""

        print("Species:")
        for species in self.species_list:
            str_species += "%s, "%species.name
        print(str_species)

    def print_reactions(self):
        print("Reactions:")
        for reaction in self.reaction_list:
            print(reaction.formula)

#####################
# Specialize Solver #
#####################
def solve_rhoT1(mixture, K_p):
    rho = mixture.rho_mix
    R_hat = mixture.R_hat
    T = mixture.T

    M_hat_all = np.zeros(mixture.n_species)
    eta_all = np.zeros(mixture.n_species)

    for idx, species in enumerate(mixture.species_list):
        M_hat_all[idx] = species.M_hat

    C = K_p/(rho*R_hat*T)

    a = 1/C
    b = M_hat_all[1]/M_hat_all[0]
    c = -1/M_hat_all[0]

    eta_all[1] = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    eta_all[0] = (1 - M_hat_all[1]*eta_all[1])/M_hat_all[0]

    return eta_all

def solve_pT_RG8_Newton(p_0_all, p_mix, K_p_all, ratio_NO,
    omega_min=0.2, max_iter=5000, tol=1e-7, is_print=True):
    
    p_all = np.array(p_0_all)

    res = res_RG8(p_all, p_mix, K_p_all, ratio_NO).reshape((-1, 1))
    for iteration in range(max_iter):
        
        jac = res_RG8_prime(p_all, p_mix, K_p_all, ratio_NO)
        jac_inv = np.linalg.inv(jac)
        dp = -np.matmul(jac_inv, res).reshape(-1)

        # Adaptive URF updating
        #x = np.max(np.abs(dp))
        #omega = 1 - (1 - omega_min)*np.exp(-1/x)

        # Linear updating
        omega = 1

        dy = omega*dp
        for idx, p in enumerate(p_all):
            y = p_all[idx] + dy[idx]

            if y  < 0:
                p_all[idx] = abs(p_all[idx])/2
            else:
                p_all[idx] = y

        res = res_RG8(p_all, p_mix, K_p_all, ratio_NO)
        error = np.max(np.abs(res))

        if error < tol:
            break
    if is_print:
        print("Newton iterations: %i, error: %.5e"%(iteration, error))

    return p_all

def res_RG8(p, p_mix, K_p_all, ratio_NO):
    res = np.zeros(8)

    # idx| 0   1   2   3   4   5   6   7
    # sp | N2, O2, NO, N,  O,  N+, O+, e-, 

    res[0] =   p[3]**2/p[0] - K_p_all[0]
    res[1] =   p[4]**2/p[1] - K_p_all[1]
    res[2] = p[3]*p[4]/p[2] - K_p_all[2]
    res[3] = p[5]*p[7]/p[3] - K_p_all[3]
    res[4] = p[6]*p[7]/p[4] - K_p_all[4]

    res[5] = (2*p[0] + p[2] + p[3] + p[5])/\
        (2*p[1] + p[2] + p[4] + p[6]) - ratio_NO

    res[6] = p[5] + p[6] - p[7]

    res[7] = np.sum(p) - p_mix

    c = np.zeros(8)
    c[:5] = K_p_all
    c[5] = ratio_NO
    c[6] = 1
    c[7] = p_mix

    res /= c

    return res

def res_RG8_prime(p, p_mix, K_p_all, ratio_NO):
    res_prime = np.zeros((8, 8))

    # reaction equlibrium
    res_prime[0, 0] = -p[3]**2/p[0]**2
    res_prime[0, 3] = 2*p[3]/p[0]

    res_prime[1, 1] = -p[4]**2/p[1]**2
    res_prime[1, 4] = 2*p[4]/p[1]

    res_prime[2, 2] = -p[3]*p[4]/p[2]**2
    res_prime[2, 3] = p[4]/p[2]
    res_prime[2, 4] = p[3]/p[2]

    res_prime[3, 3] = -p[5]*p[7]/p[3]**2
    res_prime[3, 5] = p[7]/p[3]
    res_prime[3, 7] = p[5]/p[3]

    res_prime[4, 4] = -p[6]*p[7]/p[4]**2
    res_prime[4, 6] = p[7]/p[4]
    res_prime[4, 7] = p[6]/p[4]

    numer = 2*p[0] + p[2] + p[3] + p[5]
    denom = 2*p[1] + p[2] + p[4] + p[6]

    # equlibrium
    res_prime[5, 0] = 2/denom
    res_prime[5, 1] = -2*numer/denom**2
    res_prime[5, 2] = 1/denom - numer/denom**2
    res_prime[5, 3] = 1/denom
    res_prime[5, 4] = -numer/denom**2
    res_prime[5, 5] = 1/denom
    res_prime[5, 6] = -numer/denom**2

    res_prime[6, 5] = 1
    res_prime[6, 6] = 1
    res_prime[6, 7] = -1

    res_prime[7] = 1

    c = np.zeros(8)
    c[:5] = K_p_all
    c[5] = ratio_NO
    c[6] = 1
    c[7] = p_mix
    c = c.reshape((-1, 1))
    res_prime = res_prime/c

    return res_prime

def get_RG8_a_jacobian(mixture, dx=1e-8):

    p_all_0 = mixture.p_all

    T0 = mixture.T
    dT = T0*dx
    T1 = T0 + dT
    T2 = T0 - dT

    # FDM approach
    K_p_all_1 = mixture.compute_K_p(T1)
    K_p_all_2 = mixture.compute_K_p(T2)
    K_p_all_0 = mixture.compute_K_p(T0)

    mixture.compute_all()
    s_mix_0 = mixture.s_mix
    p_all = mixture.p_all

    dK_pdT = (K_p_all_1 - K_p_all_2)/(2*dT)
    #dsdT = (s_mix_1 - s_mix_2)/(2*dT)


    eta_all = mixture.eta_all
    rho = mixture.rho_mix
    R = mixture.R_hat
    n_s = mixture.n_species

    rhoRT = rho*R*T0

    n_Y = n_s + 2
    Jacobian = np.zeros((n_Y, n_Y))

    # reaction equlibrium
    Jacobian[0, 0] = -eta_all[3]**2/eta_all[0]**2
    Jacobian[0, 3] = 2*eta_all[3]/eta_all[0]
    Jacobian[0, -2] = K_p_all_0[0]/(rho*rhoRT)
    Jacobian[0, -1] = (K_p_all_0[0] - T0*dK_pdT[0])/(rhoRT*T0)

    Jacobian[1, 1] = -eta_all[4]**2/eta_all[1]**2
    Jacobian[1, 4] = 2*eta_all[4]/eta_all[1]
    Jacobian[1, -2] = K_p_all_0[1]/(rho*rhoRT)
    Jacobian[1, -1] = (K_p_all_0[1] - T0*dK_pdT[1])/(rhoRT*T0)

    Jacobian[2, 2] = -eta_all[3]*eta_all[4]/eta_all[2]**2
    Jacobian[2, 3] = eta_all[4]/eta_all[2]
    Jacobian[2, 4] = eta_all[3]/eta_all[2]
    Jacobian[2, -2] = K_p_all_0[2]/(rho*rhoRT)
    Jacobian[2, -1] = (K_p_all_0[2] - T0*dK_pdT[2])/(rhoRT*T0)

    Jacobian[3, 3] = -eta_all[5]*eta_all[7]/eta_all[3]**2
    Jacobian[3, 5] = eta_all[7]/eta_all[3]
    Jacobian[3, 7] = eta_all[5]/eta_all[3]
    Jacobian[3, -2] = K_p_all_0[3]/(rho*rhoRT)
    Jacobian[3, -1] = (K_p_all_0[3] - T0*dK_pdT[3])/(rhoRT*T0)

    Jacobian[4, 4] = -eta_all[6]*eta_all[7]/eta_all[4]**2
    Jacobian[4, 6] = eta_all[7]/eta_all[4]
    Jacobian[4, 7] = eta_all[6]/eta_all[4]
    Jacobian[4, -2] = K_p_all_0[4]/(rho*rhoRT)
    Jacobian[4, -1] = (K_p_all_0[4] - T0*dK_pdT[4])/(rhoRT*T0)

    # conservations
    A = 2*eta_all[0] + eta_all[2] + eta_all[3] + eta_all[5]
    B = 2*eta_all[1] + eta_all[2] + eta_all[4] + eta_all[6]
    Jacobian[5, 0] = 2/B
    Jacobian[5, 1] = -2*A/B**2
    Jacobian[5, 2] = 1/B - A/B**2
    Jacobian[5, 3] = 1/B
    Jacobian[5, 4] = -A/B**2
    Jacobian[5, 5] = 1/B
    Jacobian[5, 6] = -A/B**2

    Jacobian[6, 5] = 1
    Jacobian[6, 6] = 1
    Jacobian[6, 7] = -1

    U0 = np.zeros(n_Y)
    U0[:8] = eta_all
    U0[-2] = rho
    U0[-1] = T0

    Jacobian[9] = get_ds_mix_dU(eta_all, rho, T0, mixture, dx)

    for idx in range(n_s):
        species = mixture.species_list[idx]

        Jacobian[7, idx] = species.M_hat
        Jacobian[8, idx] = rhoRT
        Jacobian[9, idx] = species.M_hat*species.s_total - R

    Jacobian[8, -2] = np.sum(eta_all)*R*T0
    Jacobian[8, -1] = np.sum(eta_all)*rho*R

    Jacobian[9, -2] = np.sum(-eta_all**2*R**2*T0/p_all)
    #Jacobian[9, -1] = dsdT
    mixture.set_T(T0)
    return Jacobian

def get_ds_mix_dU(eta_all, rho, T, mixture, dx, n_Y=10):
    U0 = np.zeros(n_Y)
    U0[:8] = eta_all
    U0[-2] = rho
    U0[-1] = T

    dsdU = np.zeros(n_Y)
    for u_idx in range(n_Y):
        U1 = np.array(U0)
        U2 = np.array(U0)
        dy = U0[u_idx]*dx

        U1[u_idx] += dy
        s_1 = get_s_mix(U1, mixture)
        
        U2[u_idx] -= dy
        s_2 = get_s_mix(U2, mixture)

        dsdU[u_idx] = (s_1 - s_2)/(2*dy)

    return dsdU 

def get_s_mix(U, mixture):
    eta_all = U[:8]
    rho = U[-2]
    T = U[-1]

    n_s = mixture.n_species
    R_hat = mixture.R_hat

    s_mix = 0 

    mixture.set_T(T)
    for s_idx in range(n_s):
        species = mixture.species_list[s_idx]
        p_s = eta_all[s_idx]*rho*R_hat*T
        species.compute_entropy(p_s)
        s_total = species.s_total

        cs = eta_all[s_idx]*species.M_hat*s_total

        s_mix += cs

    return s_mix


def solve_rhoe_RG8_Newton(eta_0_all, T_0, rho, e, mixture,
    omega_min=0.2, max_iter=1000, tol=1e-7, is_print=True):

    eta_all = np.array(eta_0_all)
    T = T_0

    res = res_RG8_eta(eta_all, T, rho, e, mixture).reshape((-1, 1))
    for iteration in range(max_iter):
        
        jac = res_rhoe_RG8_prime(eta_all, T, rho, e, mixture)
        jac_inv = np.linalg.inv(jac)
        dU = -np.matmul(jac_inv, res).reshape(-1)

        # Adaptive URF updating
        x = np.max(np.abs(dU)) + 1e-30
        omega = 1 - (1 - omega_min)*np.exp(-1/x)

        # Linear updating
        #omega = 1

        dy = omega*dU
        for idx, p in enumerate(eta_all):
            y = eta_all[idx] + dy[idx]

            if y  < 0:
                eta_all[idx] = abs(eta_all[idx])/2
            else:
                eta_all[idx] = y

        y = T + dy[-1]
        T = y if y > 0 else abs(T)/2

        res = res_RG8_eta(eta_all, T, rho, e, mixture)
        error = np.max(np.abs(res))

        if error < tol:
            break
    if is_print:
        print("rho-e Newton iterations: %i, error: %.5e"%(iteration, error))

    return eta_all, T

def res_RG8_eta(eta, T, rho, e, mixture):
    res = np.zeros(9)

    mixture.set_T(T)
    K_p_all = mixture.compute_K_p(T)
    ratio_NO = mixture.ratio_NO
    R_hat = mixture.R_hat
    rhoRT = rho*R_hat*T
    n_s = mixture.n_species

    # idx| 0   1   2   3   4   5   6   7
    # sp | N2, O2, NO, N,  O,  N+, O+, e-, 

    res[0] =     eta[3]**2/eta[0] - K_p_all[0]/rhoRT
    res[1] =     eta[4]**2/eta[1] - K_p_all[1]/rhoRT
    res[2] = eta[3]*eta[4]/eta[2] - K_p_all[2]/rhoRT
    res[3] = eta[5]*eta[7]/eta[3] - K_p_all[3]/rhoRT
    res[4] = eta[6]*eta[7]/eta[4] - K_p_all[4]/rhoRT

    res[5] = (2*eta[0] + eta[2] + eta[3] + eta[5])/\
        (2*eta[1] + eta[2] + eta[4] + eta[6]) - ratio_NO

    res[6] = eta[5] + eta[6] - eta[7]

    for s_idx in range(n_s):
        species = mixture.species_list[s_idx]
        M_hat = species.M_hat
        c = eta[s_idx]*M_hat
        e_s = species.e_total

        res[7] += c
        res[8] += c*e_s

    res[7] -= 1
    res[8] -= e

    c = np.zeros(9)
    c[:5] = K_p_all/rhoRT
    c[5] = ratio_NO
    c[6] = 1
    c[7] = 1
    c[8] = e

    res /= c

    return res

def res_rhoe_RG8_prime(eta_all, T, rho, e, mixture, dx=1e-8):
    T0 = T
    dT = T0*dx
    T1 = T0 + dT
    T2 = T0 - dT

    # FDM approach
    K_p_all_1 = mixture.compute_K_p(T1)
    K_p_all_2 = mixture.compute_K_p(T2)
    K_p_all_0 = mixture.compute_K_p(T0)

    mixture.set_T(T0)

    dK_pdT = (K_p_all_1 - K_p_all_2)/(2*dT)

    R = mixture.R_hat
    n_s = mixture.n_species

    rhoRT = rho*R*T0

    n_Y = n_s + 1
    Jacobian = np.zeros((n_Y, n_Y))

    # reaction equlibrium
    Jacobian[0, 0] = -eta_all[3]**2/eta_all[0]**2
    Jacobian[0, 3] = 2*eta_all[3]/eta_all[0]
    Jacobian[0, -1] = (K_p_all_0[0] - T0*dK_pdT[0])/(rhoRT*T0)

    Jacobian[1, 1] = -eta_all[4]**2/eta_all[1]**2
    Jacobian[1, 4] = 2*eta_all[4]/eta_all[1]
    Jacobian[1, -1] = (K_p_all_0[1] - T0*dK_pdT[1])/(rhoRT*T0)

    Jacobian[2, 2] = -eta_all[3]*eta_all[4]/eta_all[2]**2
    Jacobian[2, 3] = eta_all[4]/eta_all[2]
    Jacobian[2, 4] = eta_all[3]/eta_all[2]
    Jacobian[2, -1] = (K_p_all_0[2] - T0*dK_pdT[2])/(rhoRT*T0)

    Jacobian[3, 3] = -eta_all[5]*eta_all[7]/eta_all[3]**2
    Jacobian[3, 5] = eta_all[7]/eta_all[3]
    Jacobian[3, 7] = eta_all[5]/eta_all[3]
    Jacobian[3, -1] = (K_p_all_0[3] - T0*dK_pdT[3])/(rhoRT*T0)

    Jacobian[4, 4] = -eta_all[6]*eta_all[7]/eta_all[4]**2
    Jacobian[4, 6] = eta_all[7]/eta_all[4]
    Jacobian[4, 7] = eta_all[6]/eta_all[4]
    Jacobian[4, -1] = (K_p_all_0[4] - T0*dK_pdT[4])/(rhoRT*T0)

    # conservations
    A = 2*eta_all[0] + eta_all[2] + eta_all[3] + eta_all[5]
    B = 2*eta_all[1] + eta_all[2] + eta_all[4] + eta_all[6]
    Jacobian[5, 0] = 2/B
    Jacobian[5, 1] = -2*A/B**2
    Jacobian[5, 2] = 1/B - A/B**2
    Jacobian[5, 3] = 1/B
    Jacobian[5, 4] = -A/B**2
    Jacobian[5, 5] = 1/B
    Jacobian[5, 6] = -A/B**2

    Jacobian[6, 5] = 1
    Jacobian[6, 6] = 1
    Jacobian[6, 7] = -1

    for idx in range(n_s):
        species = mixture.species_list[idx]
        M_hat = species.M_hat
        e_s = species.e_total
        cv_s = species.cv_total
        Jacobian[7, idx] = M_hat
        Jacobian[8, idx] = M_hat*e_s
        Jacobian[8, -1] += eta_all[idx]*M_hat*cv_s

    Jacobian[:5] /= K_p_all_0.reshape((-1, 1))/rhoRT
    Jacobian[5] /= mixture.ratio_NO
    Jacobian[6] /= 1
    Jacobian[7] /= 1
    Jacobian[8] /= e

    return Jacobian

def solve_rhoe_RG8_lowtemp(eta_0_all, T_0, rho, e, mixture,
    omega_min=0.2, max_iter=100, tol=1e-7, is_print=True):
    n_s = mixture.n_species
    T = np.array(T_0)
    eta_all = np.array(eta_0_all)
    
    R_hat = mixture.R_hat
    ratio_NO = mixture.ratio_NO

    M_hat_all = np.zeros(n_s)
    for s_idx in range(n_s):
        species = mixture.species_list[s_idx]
        M_hat_all[s_idx] = species.M_hat


    for iteration in range(max_iter):   
        mixture.set_T(T)
        K_p_all = mixture.compute_K_p(T)
        rhoRT = rho*R_hat*T

        for j in range(max_iter):
            # idx| 0   1   2   3   4   5   6   7
            # sp | N2, O2, NO, N,  O,  N+, O+, e-, 

            # N_2, O_2 => N, O
            eta_all[3] = np.sqrt(K_p_all[0]/rhoRT*eta_all[0])
            eta_all[4] = np.sqrt(K_p_all[1]/rhoRT*eta_all[1])

            # N, O => NO
            eta_all[2] = eta_all[3]*eta_all[4]/(K_p_all[2]/rhoRT) 

            # N, O => Np, Op, em
            c1 = eta_all[3]*K_p_all[3]/rhoRT
            c2 = eta_all[4]*K_p_all[4]/rhoRT

            eta_all[5] = c1/np.sqrt(c1 + c2)
            eta_all[6] = c2/np.sqrt(c1 + c2)
            eta_all[7] = eta_all[5] + eta_all[6]

            # rest => N_2, O_2
            a = eta_all[2] + eta_all[3] + eta_all[5]
            b = eta_all[2] + eta_all[4] + eta_all[6]
            c = ratio_NO
            d = 1 - np.sum(M_hat_all[2:]*eta_all[2:])

            k1 = 2 + 2*c*M_hat_all[0]/M_hat_all[1]
            k2 = 2*c*d/M_hat_all[1] + b*c - a

            eta_all[0] = k2/k1
            eta_all[1] = (d - M_hat_all[0]*eta_all[0])/M_hat_all[1]

            res_N_2 = eta_all[3]**2/eta_all[0]/(K_p_all[0]/rhoRT) - 1
            if abs(res_N_2) < tol:
                break

        c_all = M_hat_all*eta_all
        e_mix = 0
        cv_mix = 0
        for s_idx in range(n_s):
            species = mixture.species_list[s_idx]
            e_mix += c_all[s_idx]*species.e_total
            
            #print(species.T, species.e_total)
            cv_mix += c_all[s_idx]*species.cv_total
        
        de = e_mix - e
        dT = -de/cv_mix

        T = T + dT
        res_0 = res_RG8_eta(eta_all, T, rho, e, mixture)

        if np.abs(dT/T) < tol:
            break

    
    res_0 = res_RG8_eta(eta_all, T, rho, e, mixture)
    error = np.max(np.abs(res_0))

    if is_print:
        print("rho-e lowtemp iterations: %i, error: %.5e"%(iteration, error))

    return eta_all, T
        
    # 

if __name__ == "__main__":
    pass