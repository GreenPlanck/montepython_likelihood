import os
import numpy as np
from montepython.likelihood_class import Likelihood



from act_dr6_lenslike import ACTDR6LensLike


class act_dr6_lenslike(Likelihood):


    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        self.act = ACTDR6LensLike(
                    {'lens_only':self.lens_only,
                    'stop_at_error':self.stop_at_error,
                    'lmax':self.lmax,
                    'variant':self.variant},
                    packages_path=self.packages_path)

        self.need_cosmo_arguments(data, {'l_max_scalars': self.lmax})
        print("initial finished")
        print("=================")

    # noinspection PyUnboundLocalVariable
    def loglkl(self, cosmo, data):
        cls = self.get_cl(cosmo)
        
        return self.act.loglike(cls)
