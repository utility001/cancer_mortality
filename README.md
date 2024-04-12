# OLS Regression Challenge: Predicting Cancer Mortality Rates for US Counties

## Description
The aim of this project is to predict cancer mortality rates for different counties in the US using multivariate multivariate Ordinary Least Squares (OLS) regression model.  
The dataset was aggregated from various sources, including the American Community Survey (census.gov), clinicaltrials.gov, and cancer.gov
The description of the various feature in the dataset is located in the [data description file](data_description.md)

## Deliverables
**Model selection**:
A lasso regression model was used to train the dataset.

**Exploratory data analysis**  
The dataset was explored for the purposes of understanding and identifying various errors  
Outlier detection and removal were also performed  
One hot encoding was performed on the categorical columns  

**Featue selection**
The F-statistic scoring function of the SelectKBest module from Sklearn library was used to perform feature selection.

**Most important features**
The most important features include. 
+ Incidence rate
+ Median income
+ percentage of the population that are covered medically (public or private coverage),
+ percentage of unempoyment, etc

**Metrics**: 
+ Root Mean Squared Error was selected as the preffered metric for this specific problem statemetn

**Model Diagnostics**: 
   - Linearity and multicolinearity of model parameters. - The linearity and multicolinearity assumption is satisfied as lasso regression was used for automatic feature selection
   - Serial independence of errors and  homoskedasticity were satisfied but there seems to be some specific outliers in plot
   - The residuals were also verified to be normal

## Note
This project is a work in progress and various non-linear models will be used to get better accuracy in the future