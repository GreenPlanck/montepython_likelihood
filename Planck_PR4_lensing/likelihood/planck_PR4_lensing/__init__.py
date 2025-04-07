# https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/sn/pantheonplus.py
import os
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
_twopi = 2 * np.pi

import planckpr4lensing
#self.pr4 = planckpr4lensing.planckpr4lensing.PlanckPR4Lensing()

def chi_squared(c_inv, delta):
    """
    Compute chi squared, i.e. delta.T @ c_inv @ delta

    :param c_inv: symmetric positive definite inverse covariance matrix
    :param delta: 1D array
    :return: delta.T @ c_inv @ delta
    """
    if len(delta) < 1500:
        return c_inv.dot(delta).dot(delta)
    else:
        # use symmetry
        return scipy.linalg.blas.dsymv(alpha=1.0,
                                       a=c_inv if np.isfortran(c_inv) else c_inv.T,
                                       x=delta, lower=0).dot(delta)
    

class planck_PR4_lensing(Likelihood,planckpr4lensing.planckpr4lensing.PlanckPR4Lensing):

    _fast_chi_squared = staticmethod(chi_squared)

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        planckpr4lensing.planckpr4lensing.PlanckPR4Lensing.__init__(self,{'dataset_file': os.path.join(self.data_directory, self.dataset_file)})

        print("initial finished")
        print("=================")



    def get_theory_map_cls(self, Cls, data_params=None):
        for i in range(self.nmaps_required):
            for j in range(i + 1):
                CL = self.map_cls[i, j]
                combination = "".join([self.field_names[k] for k in CL.theory_ij]).lower()
                cls = Cls.get(combination)
                if cls is not None:
                    CL.CL[:] = cls[self.pcl_lmin:self.pcl_lmax + 1]
                else:
                    CL.CL[:] = 0
        self.adapt_theory_for_maps(self.map_cls, data_params or {})

    def adapt_theory_for_maps(self, cls, data_params):
        if self.aberration_coeff:
            self.add_aberration(cls)
        self.add_foregrounds(cls, data_params)
        if self.calibration_param is not None and self.calibration_param in data_params:
            for i in range(self.nmaps_required):
                for j in range(i + 1):
                    CL = cls[i, j]
                    if CL is not None:
                        if CL.theory_ij[0] <= 2 and CL.theory_ij[1] <= 2:
                            CL.CL /= data_params[self.calibration_param] ** 2

    def add_foregrounds(self, cls, data_params):
        pass


    def elements_to_matrix(self, X, M):
        ix = 0
        for i in range(self.nmaps):
            M[i, 0:i] = X[ix:ix + i]
            M[0:i, i] = X[ix:ix + i]
            ix += i
            M[i, i] = X[ix]
            ix += 1



    # noinspection PyUnboundLocalVariable
    def loglkl(self, cosmo, data):
        r"""
        Get log likelihood from the dls (CMB C_l scaled by L(L+1)/2\pi)

        :param dls: dictionary of d_l ('tt', etc)
        :param data_params: likelihood nuisance parameters
        :return: log likelihood
        """
        cls = self.get_cl(cosmo)
        fac = cls['ell'] * (cls['ell']+1) / (2*np.pi)
        cmb_typ = ['tt','te','ee','pp','tp','bb']
        dls = {mode:np.zeros_like(fac) for mode in cmb_typ}
        for mode in cmb_typ:
            if mode == 'pp': dls[mode][cls['ell']] = (cls['ell'] * (cls['ell']+1))**2 / (2*np.pi)*cls[mode]
            elif mode == 'tp' or mode == 'ep': dls[mode][cls['ell']] = (cls['ell'] * (cls['ell']+1))**(3./2.) / (2*np.pi)*cls[mode]
            else: dls[mode][cls['ell']] = fac*cls[mode]
        data_params = {par:data.mcmc_parameters[par]['current'] for par in data.get_mcmc_parameters(['nuisance'])}
        self.get_theory_map_cls(dls, data_params)
        C = np.empty((self.nmaps, self.nmaps))
        big_x = np.empty(self.nbins_used * self.ncl_used)
        vecp = np.empty(self.ncl)
        chisq = 0
        if self.binned:
            binned_theory = self.get_binned_map_cls(self.map_cls)
        else:
            Cs = np.zeros((self.nbins_used, self.nmaps, self.nmaps))
            for i in range(self.nmaps):
                for j in range(i + 1):
                    CL = self.map_cls[i, j]
                    if CL is not None:
                        Cs[:, i, j] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
                        Cs[:, j, i] = CL.CL[self.bin_min - self.pcl_lmin:
                                            self.bin_max - self.pcl_lmin + 1]
        for b in range(self.nbins_used):
            if self.binned:
                self.elements_to_matrix(binned_theory[b, :], C)
            else:
                C[:, :] = Cs[b, :, :]
            if self.cl_noise is not None:
                C += self.noise_matrix[b]
            if self.like_approx == 'exact':
                chisq += self.exact_chi_sq(
                    C, self.bandpower_matrix[b], self.bin_min + b)
                continue
            elif self.like_approx == 'HL':
                try:
                    self.transform(
                        C, self.bandpower_matrix[b], self.fiducial_sqrt_matrix[b])
                except np.linalg.LinAlgError:
                    self.log.debug("Likelihood computation failed.")
                    return -np.inf
            elif self.like_approx == 'gaussian':
                C -= self.bandpower_matrix[b]
            self.matrix_to_elements(C, vecp)
            big_x[b * self.ncl_used:(b + 1) * self.ncl_used] = vecp[
                self.cl_used_index]
        if self.like_approx == 'exact':
            return -0.5 * chisq
        return -0.5 * self._fast_chi_squared(self.covinv, big_x)

    @staticmethod
    def transform(C, Chat, Cfhalf):
        # HL transformation of the matrices
        if C.shape[0] == 1:
            rat = Chat[0, 0] / C[0, 0]
            C[0, 0] = (np.sign(rat - 1) *
                       np.sqrt(2 * np.maximum(0, rat - np.log(rat) - 1)) *
                       Cfhalf[0, 0] ** 2)
            return
        diag, U = np.linalg.eigh(C)
        rot = U.T.dot(Chat).dot(U)
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        U.dot(rot.dot(U.T), rot)
        diag, rot = np.linalg.eigh(rot)
        diag = np.sign(diag - 1) * np.sqrt(2 * np.maximum(0, diag - np.log(diag) - 1))
        Cfhalf.dot(rot, U)
        for i, d in enumerate(diag):
            rot[:, i] = U[:, i] * d
        rot.dot(U.T, C)
