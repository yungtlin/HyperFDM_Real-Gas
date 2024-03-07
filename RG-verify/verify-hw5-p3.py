###########
# Modules #
###########
import numpy as np

import sys
sys.path.append("../libs/real-gas/")
from Species import Monatomic, Diatomic
from Mixture import Reaction, Mixture
from gas_models import get_model_hw5

if __name__ == "__main__":
    model = get_model_hw5()

    weights = np.array([1, 0])
    model.init_composition(weights)
    #model.print_info()

    rho = 1
    T = 5000

    model.compute_rhoT1(rho, T)

    p_answer = np.array([1.478e6, 1.326e4])
    error = (model.p_all - p_answer)/np.abs(p_answer)

    print(error)
