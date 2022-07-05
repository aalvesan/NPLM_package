# NPLM_MUSiC
Adapatation of the New Physics Learning Machine (NPLM) algorithm to MUSiC analysis at CMS.

## Short description:
Model Unspecifc Searches in CMS (MUSiC) is an analysis strategy looking for discrepancies between the datasets provided by the CMS expriment at the LHC in Switzerland and Monte Carlo generated datasets following the theoretically predicted Standard Model. MUSiC does not assume the nature of such discrepencies, and thus remains independent of theorical new physics hyphotheses. If MUSiC finds an anomaly in the CMS data, it can be a sign of new physics.

Since NPLM is also a strategy to detect data departures from a given reference model, with no prior bias on the nature of the new physics model, it is reasonable to expand MUSiC to make use of the NPLM algorithm. NPLM employs neural networks, leveraging their virtues as flexible function approximants, but builds its foundations directly on the canonical likelihood-ratio approach to hypothesis testing. The algorithm compares observations with an auxiliary set of reference-distributed events, possibly obtained with a Monte Carlo event generator. It returns a p-value, which measures the compatibility of the reference model with the data. It also identifies the most discrepant phase-space region of the dataset, to be selected for further investigation. Imperfections due to mis-modelling in the reference dataset can be taken into account straightforwardly as nuisance parameters.

## MUSiC related works:
- *"..."* (citation)

## NPLM related works:
- *"Learning New Physics from a Machine"* ([Phys. Rev. D](https://doi.org/10.1103/PhysRevD.99.015014))
- *"Learning Multivariate New Physics"* ([Eur. Phys. J. C](https://doi.org/10.1140/epjc/s10052-021-08853-y))
- *"Learning New Physics from an Imperfect Machine"* ([Eur. Phys. J. C](https://doi.org/10.1140/epjc/s10052-022-10226-y))

## NPLM envirnoment set up:
Create a virtual environment with the packages specified in `requirements.txt`
  ```
  python3 -m venv env
  source env/bin/activate
  ```
  to be sure that pip is up to date
  ```
  pip install --upgrade pip
  ```
  install the packaes listed in `requirements.txt`
  ```
  pip install -r requirements.txt 
  ```
  to see what you installed (check if successful)
  ```
  pip freeze
  ```
  Now you are ready to download the [NPLM](https://pypi.org/project/NPLM/) package:
  ```
  pip install NPLM
  ```
## Envirnoment set up on lxplus at Cern
  Just source the virtual environment: 
  ```
  source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc10-opt/setup.sh
  ```
  Download the [NPLM](https://pypi.org/project/NPLM/) package:
  ```
  pip install NPLM
  ```
