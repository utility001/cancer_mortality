### Description
Predict cancer mortality rates for US counties.

### Background
These data were aggregated from a number of sources including the American Community Survey (census.gov), clinicaltrials.gov, and cancer.gov. Most of the data preparation process can be viewed here.

### Your Task
Build a multivariate Ordinary Least Squares regression model to predict "TARGET_deathRate".

### Deliverables
a. Your final model equation  
b. The statistical software output including (adjusted) R-squared and Root Mean Squared Error (RMSE)  
c. Your code file (if you used a programming language)  
d. Model diagnostics including statistics and visualizations:  
   - Assess linearity of model (parameters)  
   - Assess serial independence of errors  
   - Assess heteroskedasticity  
   - Assess normality of residual distribution  
   - Assess multicollinearity  
e. Your interpretation of the model  
f. Other factors to consider:  
   - Are there any outliers?  
   - Are there missing values?  
   - How will you handle categorical variables?  


### Data description

| Variable                   | Description                                                                                            | 
|----------------------------|--------------------------------------------------------------------------------------------------------| 
| TARGET_deathRate           | Dependent variable. Mean per capita (100,000) cancer mortalities (years 2010-2016)                    | 
| avgAnnCount                | Mean number of reported cases of cancer diagnosed annually (years 2010-2016)                            | 
| avgDeathsPerYear           | Mean number of reported mortalities due to cancer (years 2010-2016)                                     | 
| incidenceRate              | Mean per capita (100,000) cancer diagnoses (years 2010-2016)                                            | 
| medianIncome               | Median income per county (2013 Census Estimates)                                                        | 
| popEst2015                 | Population of county (2013 Census Estimates)                                                             | 
| povertyPercent             | Percent of populace in poverty (2013 Census Estimates)                                                  | 
| studyPerCap                | Per capita number of cancer-related clinical trials per county (years 2010-2016)                        | 
| binnedInc                  | Median income per capita binned by decile (2013 Census Estimates)                                        | 
| MedianAge                  | Median age of county residents (2013 Census Estimates)                                                   | 
| MedianAgeMale              | Median age of male county residents (2013 Census Estimates)                                              | 
| MedianAgeFemale            | Median age of female county residents (2013 Census Estimates)                                            | 
| Geography                  | County name (2013 Census Estimates)                                                                     | 
| AvgHouseholdSize           | Mean household size of county (2013 Census Estimates)                                                    | 
| PercentMarried             | Percent of county residents who are married (2013 Census Estimates)                                      | 
| PctNoHS18_24               | Percent of county residents ages 18-24 highest education attained: less than high school (2013 Census Estimates) | 
| PctHS18_24                 | Percent of county residents ages 18-24 highest education attained: high school diploma (2013 Census Estimates) | 
| PctSomeCol18_24           | Percent of county residents ages 18-24 highest education attained: some college (2013 Census Estimates) | 
| PctBachDeg18_24           | Percent of county residents ages 18-24 highest education attained: bachelor's degree (2013 Census Estimates) | 
| PctHS25_Over               | Percent of county residents ages 25 and over highest education attained: high school diploma (2013 Census Estimates) | 
| PctBachDeg25_Over         | Percent of county residents ages 25 and over highest education attained: bachelor's degree (2013 Census Estimates) | 
| PctEmployed16_Over        | Percent of county residents ages 16 and over employed (2013 Census Estimates)                           | 
| PctUnemployed16_Over      | Percent of county residents ages 16 and over unemployed (2013 Census Estimates)                         | 
| PctPrivateCoverage        | Percent of county residents with private health coverage (2013 Census Estimates)                        | 
| PctPrivateCoverageAlone   | Percent of county residents with private health coverage alone (no public assistance) (2013 Census Estimates) | 
| PctEmpPrivCoverage        | Percent of county residents with employee-provided private health coverage (2013 Census Estimates)     | 
| PctPublicCoverage         | Percent of county residents with government-provided health coverage (2013 Census Estimates)           | 
| PctPubliceCoverageAlone   | Percent of county residents with government-provided health coverage alone (2013 Census Estimates)     | 
| PctWhite                  | Percent of county residents who identify as White (2013 Census Estimates)                              | 
| PctBlack                  | Percent of county residents who identify as Black (2013 Census Estimates)                              | 
| PctAsian                  | Percent of county residents who identify as Asian (2013 Census Estimates)                              | 
| PctOtherRace              | Percent of county residents who identify in a category which is not White, Black, or Asian (2013 Census Estimates) | 
| PctMarriedHouseholds      | Percent of married households (2013 Census Estimates)                                                   | 
| BirthRate                 | Number of live births relative to the number of women in the county (2013 Census Estimates)             | 