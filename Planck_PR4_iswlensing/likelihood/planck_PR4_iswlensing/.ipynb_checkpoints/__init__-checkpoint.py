import os
import numpy as np
from montepython.likelihood_class import Likelihood
from copy import deepcopy
#from . import lik
from planckpr4lensing.iswlens_jtliks import lik
npipe_prefix = 'jtlik_data_lmax99_PR4_July25'
PR3_T, verbose = True, False


class planck_PR4_iswlensing(Likelihood,lik.cobaya_jtlikPRXpp):

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        lik.cobaya_jtlikPRXpp.__init__(self,X=4, prefix=npipe_prefix, cobaya_type=self.type, force_PR3TT=PR3_T, verbose=verbose)


    def new_logp(self, Cls,**params_values):
        if self.calibration_param in params_values: #FIXME: what to do with PP and PT ? looks like the public lensing lik do nothing
            Cls = deepcopy(Cls) # not sure if this is needed
            for s in ['tt', 'te', 'et', 'ee']:
                if s in Cls.keys():
                    Cls[s] /= params_values[self.calibration_param] ** 2
        Cls['pt']=Cls['tp']
        chi2 = getattr(self, 'get_chi2_' + self.cobaya_type)(Cls)
        return -0.5 * chi2

    def loglkl(self, cosmo, data):
        #Cls = self.provider.get_Cl(ell_factor=False, units='FIRASmuK2')
        Cls = self.get_cl(cosmo)
        params_values={par:data.mcmc_parameters[par]['current']*data.mcmc_parameters[par]['scale'] for par in data.get_mcmc_parameters(['nuisance'])}
        return self.new_logp(Cls,**params_values)
        