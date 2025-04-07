# https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/sn/pantheonplus.py
import os
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
_twopi = 2 * np.pi

class planck_PR4_lensing(Likelihood):

    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)
        self._read_data_file(os.path.join(self.data_directory, self.data_file))
        self.covs = {}
        for name in ['mag']:
            #self.log.debug('Reading covmat for: %s ' % name)
            self.covs[name] = self._read_covmat(
                os.path.join(self.data_directory, self.covmat_file))
        self.alphabeta_covmat = False
        self.configure()
        self.inverse_covariance_matrix()
        if not self.use_abs_mag:
            self._marginalize_abs_mag()
        self.marginalize = False

    def _apply_mask(self, zmask):
        for col in self.cols:
            setattr(self, col, getattr(self, col)[zmask])
        for name, cov in self.covs.items():
            self.covs[name] = cov[np.ix_(zmask, zmask)]


    def configure(self):
        self.pre_vars = self.mag_err ** 2

    def _read_cols(self, data_file, file_cols, sep=None):
        #self.log.debug('Reading %s' % data_file)
        with open(data_file, 'r') as f:
            lines = f.readlines()
            line = lines[0]
            if line.startswith('#'):
                line = line[1:]
            cols = [col.strip().lower() for col in line.split(sep)]
            assert cols[0].isalpha()
            indices = [cols.index(col) for col in file_cols]
            zeros = np.zeros(len(lines) - 1)
            for col in self.cols:
                setattr(self, col, zeros.astype(dtype='f8', copy=True))
            for ix, line in enumerate(lines[1:]):
                vals = [val.strip() for val in line.split(sep)]
                vals = [vals[i] for i in indices]
                for i, (col, val) in enumerate(zip(self.cols, vals)):
                    tmp = getattr(self, col)
                    tmp[ix] = np.asarray(val, dtype=tmp.dtype)
        self.nsn = ix + 1
        print('Number of SN read: %s ' % self.nsn)
        #self.log.debug('Number of SN read: %s ' % self.nsn)
    

    def _read_data_file(self, data_file):
        file_cols = ['zhd', 'zhel', 'mu', 'muerr_final']
        self.cols = ['zcmb', 'zhel', 'mag', 'mag_err']
        self._read_cols(data_file, file_cols, sep=',')

    def _read_covmat(self, filename):
        cov = np.loadtxt(filename)
        if np.isscalar(cov[0]) and cov[0] ** 2 + 1 == len(cov):
            cov = cov[1:]
        return cov.reshape((self.nsn, self.nsn))


    def inverse_covariance_matrix(self, alpha=0, beta=0):
        if 'mag' in self.covs:
            invcovmat = self.covs['mag'].copy()
        else:
            invcovmat = 0
        if self.alphabeta_covmat:
            if np.isclose(alpha, self._last_alpha) and np.isclose(beta, self._last_beta):
                return self.invcov
            self._last_alpha = alpha
            self._last_beta = beta
            alphasq = alpha * alpha
            betasq = beta * beta
            alphabeta = alpha * beta
            if 'stretch' in self.covs:
                invcovmat += alphasq * self.covs['stretch']
            if 'colour' in self.covs:
                invcovmat += betasq * self.covs['colour']
            if 'mag_stretch' in self.covs:
                invcovmat += 2 * alpha * self.covs['mag_stretch']
            if 'mag_colour' in self.covs:
                invcovmat -= 2 * beta * self.covs['mag_colour']
            if 'stretch_colour' in self.covs:
                invcovmat -= 2 * alphabeta * self.covs['stretch_colour']
            delta = (self.pre_vars + alphasq * self.stretch_var +
                     betasq * self.colour_var + 2.0 * alpha * self.cov_mag_stretch +
                     -2.0 * beta * self.cov_mag_colour +
                     -2.0 * alphabeta * self.cov_stretch_colour)
        else:
            delta = self.pre_vars
        np.fill_diagonal(invcovmat, invcovmat.diagonal() + delta)
        self.invcov = np.linalg.inv(invcovmat)
        return self.invcov


    def _marginalize_abs_mag(self):
        deriv = np.ones_like(self.mag)[:, None]
        derivp = self.invcov.dot(deriv)
        fisher = deriv.T.dot(derivp)
        self.invcov = self.invcov - derivp.dot(np.linalg.solve(fisher, derivp.T))



    def alpha_beta_logp(self, lumdists, Mb=0., **kwargs):
        if self.use_abs_mag:
            estimated_scriptm = Mb + 25
        else:
            estimated_scriptm = 0.
        diffmag = self.mag - lumdists - estimated_scriptm
        return - diffmag.dot(self.invcov).dot(diffmag) / 2.


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
        

    def loglkl(self, cosmo, data):
        r"""
        Get log likelihood from the dls (CMB C_l scaled by L(L+1)/2\pi)

        :param dls: dictionary of d_l ('tt', etc)
        :param data_params: likelihood nuisance parameters
        :return: log likelihood
        """
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


