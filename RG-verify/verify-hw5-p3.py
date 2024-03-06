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

    model.print_info()

    rho = 1
    T = 5000
