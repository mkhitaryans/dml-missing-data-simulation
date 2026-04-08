# dml-missing-data-simulation
Double Machine Learning under model misspecification

This project replicates and extends the simulation study from Kang and Schafer (2007), which examines the performance of estimators under simultaneous misspecification of outcome and propensity models.

I implement from scratch:

Non-doubly robust estimators (IPW, stratification, OLS)
Doubly robust estimators (BC-OLS, WLS, propensity-covariate regression)
A full Double Machine Learning (DML) pipeline with cross-fitting

The extension introduces:

Short-stacking via constrained NNLS across 10 learners (OLS, Lasso, Random Forests, and neural networks)
Cross-fitted nuisance estimation for both outcome and propensity models
Simulation-based evaluation under severe model misspecification

Results are benchmarked against the DoubleML package to validate correctness and robustness.
