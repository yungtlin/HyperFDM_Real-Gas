###########
# Modules #
###########
import numpy as np

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
        
        # O/N
        self.ratio_all = self.get_ratio(self.species_list, self.x0_all)

    def get_ratio(self, species_list, c):
        return c[1]/c[0]

    ###########
    # Compute #
    ###########
    def compute_pT(self, p, T):
        self.set_T(T)

        K_p_O2, K_p_N2 = self.compute_K_p(T)

        self.p_all = solve_constant_p(p, self.ratio_all, K_p_O2, K_p_N2)
        self.compute_all()

    def compute_rhoT1(self, rho, T):
        self.rho_mix = rho
        self.set_T(T)

        R_hat = self.species_list[0].R_hat
        K_p_all = self.compute_K_p(T)

        eta_all = solve_rhoT1(self, K_p_all[0])

        self.compute_all_eta(eta_all)
        self.rho_mix = rho

    def compute_K_p(self, T):
        
        K_p_all = np.zeros(self.n_reaction)
        for idx, reaction in enumerate(self.reaction_list):
            reaction.compute_K_p(T)
            K_p_all[idx] = reaction.K_p

        return K_p_all

    # with p_all
    def compute_all(self):
        self.p_mix = np.sum(self.p_all)
        self.x_all = self.p_all/self.p_mix

        xM_all = np.zeros(self.n_species)
        e_all = np.zeros(self.n_species)

        for idx, species in enumerate(self.species_list):
            xM_all[idx] = self.x_all[idx]*species.M_hat
            e_all[idx] = species.e_total

        self.c_all = xM_all/np.sum(xM_all)
        self.R_mix = species.R_hat/np.sum(xM_all)
        
        self.e_mix = np.sum(self.c_all*e_all)
        self.h_mix = self.e_mix + self.R_mix*self.T

        self.rho_mix = self.p_mix/(self.R_mix*self.T)

    def compute_all_eta(self, eta_all):
        R_hat = self.species_list[0].R_hat
        self.p_all = self.rho_mix*R_hat*self.T*eta_all

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

if __name__ == "__main__":
    pass