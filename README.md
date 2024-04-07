# OLS Regression Challenge: Predicting Cancer Mortality Rates for US Counties

## Description
The aim of this project is to predict cancer mortality rates for US countries using multivariate multivariate Ordinary Least Squares (OLS) regression model.  
The dataset was aggregated from various sources, including the American Community Survey (census.gov), clinicaltrials.gov, and cancer.gov
The description of the various feature in the dataset is located in the [data description file](data_description.md)

## Deliverables
**Model selection**:
A lasso regression model was used to train the dataset.

**Exploratory data analysis**
The dataset was explored for the purposes of understanding and identifying various errors
Outlier detection and removal were also performed
One hot encoding was performed on the categorical columns

**Most important features**
The most important features include. 
Incidence rate, Median income, percentage of the population that are covered medically (public or private coverage), percentage of unempoyment, etc

b. **Metrics**: 
The mean absolute error is `14.519` on the train test and 15.277591827778762 on the test set
The root mean squared error is: `19.500` on the train  set and `20.42` on the test st
The R2 score `19.500` on the train set and `20.42` on the test set

> Note the error is per 100_000 i.e X errors per 100_000 people in the population we are referring to

d. **Model Diagnostics**: 
   - Linearity and multicolinearity of model parameters. - The linearity and multicolinearity assumption is satisfied as lasso regression was used for automatic feature selection
   - Serial independence of errors and  homoskedasticity were satisfied
   - The residuals were also verified to be normal

## Conclusion
We were able to build a robust model to predict cancer mortailty rate in US countries