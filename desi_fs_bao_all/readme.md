A MontePython wrapper for the DESI DR1 full-shape and BAO likelihood, based on the public [DESI cosmological likelihoods repository](https://github.com/cosmodesi/desi-kp-cosmological-likelihoods/tree/main).

This package implements the `desi_fs_bao_all` likelihood for MontePython and CLASS. At matched parameter points, MontePython+CLASS gives a small difference of approximately $\Delta\chi^2 \sim 0.5$ relative to the public Cobaya+CAMB implementation.

To use this likelihood, first install and configure the original Cobaya implementation, dependencies and data (see details here https://github.com/cosmodesi/desi-kp-cosmological-likelihoods/tree/main/dr1/cobaya). Then install the MontePython wrapper and run it as a standard MontePython likelihood.