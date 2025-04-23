Montepython likelihood for DESI2024-BAO and SN-pantheonplus,union3,desy5

The original likelihood and data can be found in cobaya

```
https://github.com/CobayaSampler/cobaya/blob/master/cobaya/likelihoods/sn/pantheonplus.py

https://github.com/CobayaSampler/bao_data/tree/master
```

## Usage

If you want to specify the tracers included in your analysis, you can do the setting like

```
desi_2024_gaussian_bao_all.exclude = ['LRG1','LRG2']
```

By default, we include all the tracers.

## Bibtex

We would appreciate it if you cite our work

```
@article{Lu:2025gki,
    author = "Lu, Zhiyu and Simon, Th\'eo and Zhang, Pierre",
    title = "{Preference for evolving dark energy in light of the galaxy bispectrum}",
    eprint = "2503.04602",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "3",
    year = "2025"
}
```

If you use this code please cite following papers.

```
@article{Torrado:2020dgo,
    author = "Torrado, Jesus and Lewis, Antony",
    title = "{Cobaya: Code for Bayesian Analysis of hierarchical physical models}",
    eprint = "2005.05290",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    reportNumber = "TTK-20-15",
    doi = "10.1088/1475-7516/2021/05/057",
    journal = "JCAP",
    volume = "05",
    pages = "057",
    year = "2021"
}

@article{Brinckmann:2018cvx,
      author         = "Brinckmann, Thejs and Lesgourgues, Julien",
      title          = "{MontePython 3: boosted MCMC sampler and other features}",
      year           = "2018",
      eprint         = "1804.07261",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.CO",
      SLACcitation   = "%%CITATION = ARXIV:1804.07261;%%"
}
@article{Audren:2012wb,
      author         = "Audren, Benjamin and Lesgourgues, Julien and Benabed,
                        Karim and Prunet, Simon",
      title          = "{Conservative Constraints on Early Cosmology: an
                        illustration of the Monte Python cosmological parameter
                        inference code}",
      journal        = "JCAP",
      volume         = "1302",
      pages          = "001",
      doi            = "10.1088/1475-7516/2013/02/001",
      year           = "2013",
      eprint         = "1210.7183",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.CO",
      reportNumber   = "CERN-PH-TH-2012-290, LAPTH-048-12",
      SLACcitation   = "%%CITATION = ARXIV:1210.7183;%%",
}

```

**DESYSN**

```
@article{DES:2024tys,
    author = "Abbott, T. M. C. and others",
    collaboration = "DES",
    title = "{The Dark Energy Survey: Cosmology Results With \textasciitilde{}1500 New High-redshift Type Ia Supernovae Using The Full 5-year Dataset}",
    eprint = "2401.02929",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "FERMILAB-PUB-23-0821-PPD, DES-2023-805",
    month = "1",
    year = "2024"
}
```

**PanPlusSN**

```
@article{Brout:2022vxf,
    author = "Brout, Dillon and others",
    title = "{The Pantheon+ Analysis: Cosmological Constraints}",
    eprint = "2202.04077",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.3847/1538-4357/ac8e04",
    journal = "Astrophys. J.",
    volume = "938",
    number = "2",
    pages = "110",
    year = "2022"
}
```

**Union3SN**

```
@article{Rubin:2023ovl,
    author = "Rubin, David and others",
    title = "{Union Through UNITY: Cosmology with 2,000 SNe Using a Unified Bayesian Framework}",
    eprint = "2311.12098",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "11",
    year = "2023"
}
```

**DESIBAO**

```
@article{DESI:2024mwx,
    author = "Adame, A. G. and others",
    collaboration = "DESI",
    title = "{DESI 2024 VI: cosmological constraints from the measurements of baryon acoustic oscillations}",
    eprint = "2404.03002",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "FERMILAB-PUB-24-0154-PPD",
    doi = "10.1088/1475-7516/2025/02/021",
    journal = "JCAP",
    volume = "02",
    pages = "021",
    year = "2025"
}

@article{DESI:2025zgx,
    author = "Abdul Karim, M. and others",
    collaboration = "DESI",
    title = "{DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints}",
    eprint = "2503.14738",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "FERMILAB-PUB-25-0169-PPD",
    month = "3",
    year = "2025"
}
```

**planck_pr4_lensing**

```
@article{Carron:2022eyg,
    author = "Carron, Julien and Mirmelstein, Mark and Lewis, Antony",
    title = "{CMB lensing from Planck PR4~maps}",
    eprint = "2206.07773",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2022/09/039",
    journal = "JCAP",
    volume = "09",
    pages = "039",
    year = "2022"
}
```

**planck_pr4_iswlensing**

```
@article{Carron:2022eum,
    author = "Carron, Julien and Lewis, Antony and Fabbian, Giulio",
    title = "{Planck integrated Sachs-Wolfe-lensing likelihood and the CMB temperature}",
    eprint = "2209.07395",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1103/PhysRevD.106.103507",
    journal = "Phys. Rev. D",
    volume = "106",
    number = "10",
    pages = "103507",
    year = "2022"
}
```

act_dr6_lensing

```
@article{ACT:2023kun,
    author = "Madhavacheril, Mathew S. and others",
    collaboration = "ACT",
    title = "{The Atacama Cosmology Telescope: DR6 Gravitational Lensing Map and Cosmological Parameters}",
    eprint = "2304.05203",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "FERMILAB-PUB-23-206-PPD",
    doi = "10.3847/1538-4357/acff5f",
    journal = "Astrophys. J.",
    volume = "962",
    number = "2",
    pages = "113",
    year = "2024"
}

@article{ACT:2023dou,
    author = "Qu, Frank J. and others",
    collaboration = "ACT",
    title = "{The Atacama Cosmology Telescope: A Measurement of the DR6 CMB Lensing Power Spectrum and Its Implications for Structure Growth}",
    eprint = "2304.05202",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "FERMILAB-PUB-23-237-PPD, FERMILAB-PUB-23-237-PPD",
    doi = "10.3847/1538-4357/acfe06",
    journal = "Astrophys. J.",
    volume = "962",
    number = "2",
    pages = "112",
    year = "2024"
}
```

