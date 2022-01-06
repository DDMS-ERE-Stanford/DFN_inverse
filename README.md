# Thermal experiments for fractured rock characterization: theoretical analysis and inverse modeling

[![DOI](https://zenodo.org/badge/422305167.svg)](https://zenodo.org/badge/latestdoi/422305167)

Repository for data and code for the [DFN inversion project](https://arxiv.org/abs/2106.06632).

## Description of the purposes of included items

PDF or CDF in the following context refer to those of the particle breakthrough time in cross-borehole thermal experiments (CBTEs) in discrete fracture networks (DFNs).

### 1. Average PDF over 20 random realizations of the DFN

#### Included python scripts

pk\_pdf.py:
- go through each realization to save all the PDFs into a pickle file with the index of the PDFs.

merge\_20\_pdf.py:
- count how many realizations generated connected DFN, 
- average the PDFs over the connected realizations.

pair\_cd\_cdf.py:
- convert the averaged PDFs to iCDFs, 
- pair (C,D) with their corresponding iCDFs and save as dataset.

#### Included files

- C\_d.txt: 10000 pair of (C,D) in [2.5,6.5]x[1.0,1.3] with LHS sampling.
- [uncorr\_100\_20pdf.pkl](https://drive.google.com/file/d/1fanmymXZifl5P1eKOYYBqNuDAQaGXHM6/view?usp=sharing): PDFs for the connected realizations for simulations with 100 particles for CBTE with particle based simulation method.
- [uncorr\_1000\_20pdf.pkl](https://drive.google.com/file/d/1ff0ysWywsibzjxpkgmVu-5GWYSPKrvJo/view?usp=sharing): PDFs for the connected realizations for simulations with 1000 particles for CBTE with particle based simulation method.

Representative CDFs of the logarithm of breakthrough times (in seconds) of 100 or 1000 particles, for connected realizations of the DFN characterized by a given combination of the DFN parameters (C,D):

![CDF realizations](/images/CDFs-realizations.png)

### 2. Construct informative prior distribution for (C,D)

#### Included python script

- informative\_prior/prior\_CD.py: two methods to generate informative prior distributions for (C,D).

#### Included files saving the prior distributions

- informative\_prior/conn_regr.pkl saves the prior distribution generated with the DFN connectivity information. 
![prior with connectivity](/images/conn_prior_density.png)

- informative\_prior/kde.pkl is the prior distribution with kernel density estimation with field data in informative\_prior/Classeur1.csv. 
![prior with kde with field data](/images/prior.png)

- uncorr\_100\_20pdf_conn.pkl: the connected realizations numbers for all 10000 cases.

### 3. Surrogate model training and grid searching inversion

The NN surrogate model aims to replace the map: (C,D) --> iCDF

Input: (C, D), fracture density C:[2.5, 6.5], fractal dimension D: [1, 1.3];

Output: discrete iCDF with 50 discrete points.

The surrogate model training and grid searching code can be found in the following GoogleColab link: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qKGxPeAXvgoCEn5FLL4dPwNpkmKXgJOh?usp=sharing)

The training data can be downloaded from [CD\_ICDF\_corrected.pkl](https://drive.google.com/file/d/1fYmp4ZB4kXGwTHjsVg1rEi6G4g9hO7qJ/view?usp=sharing), with the input and output being C\_d: (10000, 2) and P\_cdf: (10000, 50), P\_cdf stores the averaged iCDFs, represented by a discrete iCDF with 50 points.

```python
filename = 'CD_ICDF_corrected.pkl'
with open(filename,'rb') as file:
  [C_d, P_cdf] = pkl.load(file, encoding='latin1') 
```

The model checkpoint saved can be downloaded from [ckp.pt](https://drive.google.com/file/d/1-3zh6nxl-Fci0qr6a6qoQ6YrpiD-SdMf/view?usp=sharing)

Load the saved checkpoint:
```python
check_data = torch.load(latest_folder+'/ckp.pt')
best_config = check_data['Best trial config']
best_trained_model = Net(ls=best_config['ls'], n_l=best_config['n_l']).to(device)

```

Examples of the surrogate model test cases:

![NN surrogate test cases](/images/NN_test.png)


## Acknowledgement

[Ray hyperparameter tuning](https://docs.ray.io/en/latest/tune/index.html)

Cite us with:

```latex
@article{zhou2021thermal,
  title={Thermal experiments for fractured rock characterization: theoretical analysis and inverse modeling},
  author={Zhou, Zitong and Roubinet, Delphine and Tartakovsky, Daniel M},
  journal={arXiv preprint arXiv:2106.06632},
  year={2021}
}
```
