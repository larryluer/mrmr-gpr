# mrmr_gpr

A simple workflow to build predictive models from small datasets (dozens to a few thousand) with many predictors (up to thousands), most of which being irrelevant for the prediction, therefore requiring feature selection.

It combines upstream mRMR - minimum Redundancy Maximum Relevance feature selection and Gaussian Process Regression.

It uses the mrmr-selection, scikit-learn and scitkit-optimize packages - please cite the authors of these
packages when using the code for manuscripts!

INPUT: any excel sheet with numeric columns; consider that Gaussian Process Regression, in the standard scikit-learn implementation, can handle at most a few thousand datasets.

TARGET SELECTION: select a single target.

JUSTIFICATION OF THE WORKFLOW:
Alternative workflows such as NGBoost / shap can handle many features out of the box without needing upstream feature selection. However, we prefer Gaussian Process Regression because of the smoothness requirement given by the covariance kernel and the ability to yield uncertainties.

JUSTIFICATION OF THE mRMR feature selection method
One critics of mRMR is the rejection of non-linear correlations. In a material science context, we reject this criticism based on the following argumentation: Given a dataset with much more predictors than rows, there will be many predictors which correlate among each other. Considering the causal structure of material science (atom positions cause geometrical, energetic and dynamic structure, structure causes primary properties, primary properties cause target functionality), we want to identify predictors which scale MONOTONICALLY with target functionality, that is, which represent proxies for primary properties, not for structure. Therefore, in a material science context, the rejection of FULLY nonlinear correlation may be an asset rather than an issue. Consider that only monotonically scaling features may be used for extrapolation.

Our implementation of GPR not only builds a predictive model for the surrogate and its uncertainty for a single prediction, but it also reports the uncertainty of the trend itself. This allows hypothesis testing such as "is positively correlated" or "has maximum". It is accomplished by a bootstrap method.

For details, see the manuscript by Marina Günthert et al. 2025

The authors:
- Larry Lüer, Marina Günthert
- Institute of Materials for Electronics and Energy technology
- Friedrich-Alexander Universität Erlangen-Nürnberg
- November 2025



## Setup

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
