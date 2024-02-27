## Modules ##
import numpy as np

##############
# Base Class #
##############
class Species:
    def __init__(self, species, constants):
        self.set_class_specific()

        self.set_universal_constant()
        self.set_species_prop(species)
        self.set_physical_constants(constants)

        self.compute_all()

    def set_class_specific(self):
        pass 

    def set_universal_constant(self):
        self.h = 6.625e-34      # Planck constant
        self.k = 1.381e-23      # Boltzmann constant
        self.N_hat = 6.02e23    # Avogadro number
        self.R_hat = self.N_hat*self.k  # Universal gas constant        

    def set_species_prop(self, species):
        comp, atom_number, temp = species
        self.composition = comp

        self.name = self.get_name()
        self.M_hat = atom_number*1e-3
        self.m = self.M_hat/self.N_hat    # weight per particle
        self.T = temp   # temperature

        self.R = self.k/self.m

    def set_physical_constants(self, constants):
        theta_r, theta_v, theta_d, g_0, g_1, theta_1, sigma = constants

        self.theta_r = theta_r
        self.theta_v = theta_v
        self.theta_d = theta_d
        self.g_0 = g_0
        self.g_1 = g_1
        self.theta_1 = theta_1
        self.sigma = sigma

    def set_T(self, T):
        if self.T != T:
            self.T = T
            self.compute_all()

    def get_name(self):
        name = ""
        for element in self.composition.keys():
            name += element
            nu = self.composition[element]
            if nu > 1:
                name += "%i"%nu
        return name

    #############
    # Computing #
    #############
    def compute_all(self):
        pass

    def compute_entropy(self, p):
        pass

    #######################
    # Partition functions #
    #######################
    # Q_trans = f_trans*V
    def translational_mode(self):
        m = self.m
        k = self.k
        T = self.T
        h = self.h
        R = self.R

        A = (2*np.pi*m*k*T)/(h**2)

        f_trans = np.power(A, 3/2)
        e_trans = 3/2*R*T

        return f_trans, e_trans

    def vibrational_mode(self):
        T = self.T
        theta_v = self.theta_v
        R = self.R

        Q_vib = 1/(1 - np.exp(-theta_v/T))
        e_vib = (R*theta_v)/(np.exp(theta_v/T) - 1)

        return Q_vib, e_vib

    def rotational_mode(self):
        T = self.T
        sigma = self.sigma
        theta_r = self.theta_r
        R = self.R

        Q_rot = T/(sigma*theta_r)
        e_rot = R*T

        return Q_rot, e_rot

    def electronic_mode(self):
        T = self.T
        g_0 = self.g_0
        g_1 = self.g_1
        theta_1 = self.theta_1
        R = self.R

        Q_el = g_0 + g_1*np.exp(-theta_1/T)

        a = (g_1/g_0)*np.exp(-theta_1/T)
        e_el = R*theta_1*a/(1 + a)

        return Q_el, e_el

    ###########
    # Entropy #
    ###########
    def entropy_translation(self, p):
        R = self.R
        T = self.T
        m = self.m
        h = self.h
        k = self.k

        A = (2*np.pi*m/h**2)
        s_trans = R*(5/2*np.log(T) - np.log(p) + np.log(A**(3/2)*k**(5/2)) + 5/2)
        return s_trans

    def entropy_rotation(self):
        R = self.R
        T = self.T 
        sigma = self.sigma
        theta_r = self.theta_r

        s_rot = R*(np.log(T/(sigma*theta_r)) + 1)
        return s_rot

    def entropy_vibration(self):
        R = self.R 
        T = self.T 
        theta_v = self.theta_v

        s_vib = R*(-np.log(1 - np.exp(-theta_v/T)) + (theta_v/T)/(np.exp(theta_v/T) - 1))
        return s_vib

    def entropy_electron(self):
        R = self.R
        T = self.T
        g_0 = self.g_0
        g_1 = self.g_1
        theta_1 = self.theta_1

        a = g_1/g_0*np.exp(-theta_1/T)

        s_el = R*(np.log(g_0) + np.log(1 + a) + ((theta_1/T)*a)/(1 + a))
        return s_el


#############
# Monatomic #
#############
class Monatomic(Species):
    def set_class_specific(self):
        pass 

    # [theta_d, g_0, g_1, theta_1]
    def set_physical_constants(self, constants):
        theta_d, g_0, g_1, theta_1 = constants
        self.theta_d = theta_d
        self.g_0 = g_0
        self.g_1 = g_1
        self.theta_1 = theta_1

        # mode 1: e_0 = 0
        # mode 2: e_0 = (self.theta_d*self.k)/(2*self.m)
        self.e_0 = (self.theta_d*self.k)/(2*self.m)

    def compute_all(self):
        self.f_trans, self.e_trans = self.translational_mode()
        self.Q_el, self.e_el = self.electronic_mode()

        self.fQ_int = self.f_trans*self.Q_el
        self.e_total = self.e_trans + self.e_el + self.e_0

    def compute_entropy(self, p):
        self.s_trans = self.entropy_translation(p)
        self.s_el = self.entropy_electron()

        self.s_total = self.s_trans + self.s_el

############
# Diatomic #
############
class Diatomic(Species):
    def set_class_specific(self):
        pass 

    # [theta_r, theta_v, theta_d, g_0, g_1, theta_1, sigma]
    def set_physical_constants(self, constants):
        theta_r, theta_v, theta_d, g_0, g_1, theta_1, sigma = constants

        self.theta_r = theta_r
        self.theta_v = theta_v
        self.theta_d = theta_d
        self.g_0 = g_0
        self.g_1 = g_1
        self.theta_1 = theta_1
        self.sigma = sigma

        # mode 1: e_0 = -self.theta_d*self.k/self.m
        # mode 2: e_0 = 0
        self.e_0 = 0

    def compute_all(self):
        self.f_trans, self.e_trans = self.translational_mode()
        self.Q_rot, self.e_rot = self.rotational_mode()
        self.Q_vib, self.e_vib = self.vibrational_mode()
        self.Q_el, self.e_el = self.electronic_mode()

        self.fQ_int = self.f_trans*self.Q_rot*self.Q_vib*self.Q_el
        self.e_total = self.e_trans + self.e_rot + self.e_vib + self.e_el + self.e_0

    def compute_entropy(self, p):
        self.s_trans = self.entropy_translation(p)
        self.s_rot = self.entropy_rotation()
        self.s_vib = self.entropy_vibration()
        self.s_el = self.entropy_electron()

        self.s_total = self.s_trans + self.s_rot + self.s_vib + self.s_el


if __name__ == "__main__":
    # Macro
    T = 5000

    ## Monatomic ##
    # [species name, atomic number, T]
    # [g_0, g_1, theta_1]
    comp_N = {"N": 1}
    species_N = [comp_N, 14, T]
    constants_N = [113000, 4, 0, 0]
    N = Monatomic(species_N, constants_N)

    ## Diatomic ##
    # [species name, atomic number, T]
    # [theta_r, theta_v, theta_d, g_0, g_1, theta_1, sigma]
    # sigma = 1, heteronuclear (e.g. NO)
    # sigma = 2, homonuclear (e.g. N_2)
    comp_N2 = {"N": 2}
    species_N2 = [comp_N2, 28, T]
    constants_N2 = [2.9, 3390, 113000, 1, 0, 0, 2]
    N_2 = Diatomic(species_N2, constants_N2)

    comp_O2 = {"O": 2}
    species_O2 = [comp_O2, 32, 1000]
    constants_O2 = [2.1, 2270, 59500, 3, 2, 11390, 2]
    O_2 = Diatomic(species_O2, constants_O2)

    # equilibrium consant
    k = N_2.k
    theta_D = N_2.theta_d
    K_p = 1/(k*T)*(N_2.fQ_int)/(N.fQ_int**2)*np.exp(theta_D/T)

    p = 101325

    O_2.compute_entropy(p)
    print(O_2.s_vib)
    #print("%.3e"%K_p)
