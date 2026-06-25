# mrmr_gpr

Update June 2026

Added a greedy additive feature selection step which can be run after mrmr. It can handle more than one target and will display their common predictors in a knowledge graph style. However, we highlight that this is not a MultiTask Gaussian Process; it is a set of Single Task Gaussian Processes.

The speciality of the workflow is to handle datasets having many more features (columns) than rows. This happens for example, if molecular structure is expressed as "mordred" features (a fixed set of avout 1800 Cheminformatics features). In this case, strong redundancy will always be present - many of these features describe strongly related cheminformatics properties. Therfore, we provide here a careful two-step redundancy rejection: first, mRMR is performed to retain the features with highest relevance and lowest redundancy. We typically retain an excess of features, because mRMR can only handle linear correlations. The retained features are then added one by one in a greedy process, only if they bring additional explanation of variance. This two-step procedure should guarantee that the final list of single task GP's contains only relevant features, while still allowing controllable nonlinearity.

A new dataset is included together with two Notebooks:
mrmr_greedy_gpr.ipynb: demonstrates the workflow and the visualization. For details on the data, see the manuscript by Tian Du et alk., 2026
XSpaceGroupGPR_split_summary.ipynb: A notebook to perform an X space group shuffle split, in cases where groups of features have a clustering tendency but a normal group shuffle split may be not available because the predctore are characterization features, not process conditions.


Original publication:

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

INSTRUCTIONS:
- after installation, open the jupyter notebook "mrmr_from_xlsx.ipynb"
- data is loaded from "df11_test.xlsx"
- after running the script multiple output figures are created

The authors:
- Larry Lüer, Marina Günthert, Qizhen Song
- Institute of Materials for Electronics and Energy technology
- Friedrich-Alexander Universität Erlangen-Nürnberg
- November 2025



## Setup

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
