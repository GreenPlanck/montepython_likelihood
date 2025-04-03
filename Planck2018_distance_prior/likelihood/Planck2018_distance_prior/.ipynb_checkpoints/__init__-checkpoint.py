import os
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings


class Planck2018_distance_prior(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)


        # define array for values of z and data points
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        self.type = np.array([], 'str')
        self.deleted_index = np.array([], 'int')
        
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for index,line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    # the first entry of the line is the identifier
                    this_line = line.split()
                    # insert into array if this id is not manually excluded
                                
                    if not this_line[0] in self.exclude:     
                        self.type = np.append(self.type, this_line[0])
                        self.data = np.append(self.data, float(this_line[1]))
                        self.error = np.append(self.error, float(this_line[2]))
                    else:
                        self.deleted_index = np.append(self.deleted_index, index)
                        
        print('--------------------')
        print(f'You have used {self.type} from planck likelihood')
        self.corre = np.loadtxt(os.path.join(self.data_directory, self.data_correlation_matrix))
        if self.deleted_index.size!=0:
            self.corre = np.delete(self.corre, self.deleted_index, axis=0)
            self.corre = np.delete(self.corre, self.deleted_index, axis=1)
            
        self.cov = np.outer(self.error, self.error) * self.corre

    # compute likelihood

    def loglkl(self, cosmo, data):

        chi2 = 0.
        theo = np.zeros_like(self.data)

        z_rec = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
        rs_rec = cosmo.get_current_derived_parameters(['rs_rec'])['rs_rec']
        DA_rec = cosmo.angular_distance(z_rec)
        H0 = cosmo.Hubble(0)
        Omega_m = cosmo.Omega_m()
        n_s = cosmo.n_s()
        omega_b = cosmo.omega_b()
        

        lA = (1.+z_rec)*np.pi*DA_rec/rs_rec
        R = (1.+z_rec)*DA_rec*H0*Omega_m**0.5

        theo = np.delete([R,lA,omega_b,n_s], self.deleted_index)
        # for obs_type in self.type:
        #     if obs_type == 'R':
        #         theo.append(R)
        #     elif obs_type == 'lA':
        #         theo.append(lA)
        #     elif obs_type == 'omega_b':
        #         theo.append(omega_b)
        #     elif obs_type == 'n_s':
        #         theo.append(n_s)
        #     else:
        #         raise ValueError(f"Unknown data type: {obs_type}")
        # theo = np.array(theo)

        invcov=np.linalg.inv(np.atleast_2d(self.cov))
        chi2 = (theo - self.data).dot(invcov).dot(theo - self.data)
            
        lkl = - 0.5 * chi2

        return lkl
