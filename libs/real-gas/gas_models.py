###########
# Modules #
###########
from Species import Monatomic, Diatomic
from Mixture import Reaction, Mixture

#########
# Model #
#########
def get_8_species():

    # Macro
    T = 300

    ## Monatomic ##
    # [species name, atomic number, T]
    # [g_0, g_1, theta_1]
    # Q_el is from Lec. note 2 - p34
    comp_N = {"N": 1}
    species_N = [comp_N, 14, T]
    constants_N = [56500, 4, 0, 0] 
    N = Monatomic(species_N, constants_N)

    comp_O = {"O": 1}
    species_O = [comp_O, 16, T]
    constants_O = [29750, 5, 4, 270] # Lec. note 2
    O = Monatomic(species_O, constants_O)

    comp_Np = {"N": 1, "+": 1}
    species_Np = [comp_Np, 14, T]
    constants_Np = [225500, 4, 0, 0] # Lec. note 2
    Np = Monatomic(species_Np, constants_Np)

    comp_Op = {"O": 1, "+": 1}
    species_Op = [comp_Op, 16, T]
    constants_Op = [187750, 5, 4, 270] # Lec. note 2
    Op = Monatomic(species_Op, constants_Op)

    comp_em = {"e": 1, "-": 1}
    species_em = [comp_em, 5.486e-4, T]
    constants_em = [0, 2, 0, 0] # Lec. note 3
    em = Monatomic(species_em, constants_em)

    ## Diatomic ##
    # [species name, atomic number, T]
    # [theta_r, theta_v, theta_z, g_0, g_1, theta_1, sigma]
    # sigma = 1, heteronuclear (e.g. NO)
    # sigma = 2, homonuclear (e.g. N_2)
    # Q_el is from Lec. note 2 - p38
    comp_N2 = {"N": 2}
    species_N2 = [comp_N2, 28, T]
    constants_N2 = [2.9, 3390, 0, 1, 0, 0, 2]
    N_2 = Diatomic(species_N2, constants_N2)

    comp_O2 = {"O": 2}
    species_O2 = [comp_O2, 32, T]
    constants_O2 = [2.1, 2270, 0, 3, 2, 11390, 2]
    O_2 = Diatomic(species_O2, constants_O2)

    comp_NO = {"N": 1, "O": 1}
    species_NO = [comp_NO, 30, T]
    constants_NO = [2.5, 2740, 10750, 2, 2, 174, 1]
    NO = Diatomic(species_NO, constants_NO)

    species_all = N_2, O_2, NO, N, O, Np, Op, em

    return species_all

# For computing Hw5 prob.3.3
def get_model_hw5():
    N_2, O_2, NO, N, O, Np, Op, em = get_8_species()

    # N_2 = 2N
    reactant = [[1, N_2]]
    product = [[2, N]]
    eqn_N2 = Reaction(reactant, product)

    species_list = [N_2, N]
    reaction_list = [eqn_N2]

    model_hw5 = Mixture(species_list, reaction_list)

    return model_hw5

if __name__ == "__main__":
    get_model_hw5()





