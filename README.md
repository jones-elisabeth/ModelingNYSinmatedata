# Modeling New York State Inmate Data: Factors associated with maximum security facility assignments
## Intro
This R code is created to model the New York State Inmate data retrieved from Kaggle. Models implemented include a decision tree, naive bayes, logistic regression, bagging, random forest, boosting, linear discriminant analysis, and a support vector machine. This project was completed as the final project for Big Data Infrastructures (MIS620) as part of my M.S. in Big Data Anlaytics.

## Technology
Project is created with:
* R Studio Version 1.3.1073
* R Version 4.0.2 (2020-06-22)

## Executive Summary 
The purpose of this report is to provide supporting data in regard to the multi-leveled issues within the U.S. mass incarceration system. This analysis will be specific to the logic behind maximum security facility assignments. The models created can be used by prison staff to understand their systems and make improvements where needed. The problem statement these models are centered around is as follows:

**Problem Statement**: Outside of an individual’s specific penal charge, certain determinants may cause an individual to be placed in maximum security level facilities over others.

The data used to assess this problem was retrieved from Kaggle and is called “NYS Inmates Under Custody: Beginning 2008”. The data used for these models included the variables Race/Ethnicity, Gender, Latest Admission Type, and Facility Security Level. The variable Facility Security Level was the dependent variable and consisted of a two-level factor (i.e., Maximum Security Assigned versus Maximum Security Not Assigned). Initial analyses showed the data to be slightly imbalanced toward Maximum Security Not Assigned (55%).

In order to prepare the data for analyses, variables were dummy coded, data was checked for missing values, and pre-processing was run. Pre-processing included the center, scale, zv, and correlation methods to convert the data into standardized scores, remove any zero variance columns, and remove any highly correlated columns.

Ultimately, eight models were run on the data including, Linear Discriminant Analysis (LDA), Recursive Partitioning and Decision Trees (RPART), Naïve Bayes (NB), Logistic Regression, Bagging, Random Forest (RF), Boosting, and Support Vector Machine (SVM). Most of the models had similar results in their performance measures. The testing ROC for all models fell between 0.553 to 0.570. The test True Positive Rate was most often low, with six out of the eight models scoring 6% or below, however two of the eight reached a 51% True Positive Rate. The True Negative Rate was generally high ranging from 61% to over 99% (Table 1). 

Table 1: Testing Performance Measures
| Model    | Accuracy (ROC) | Balanced Accuracy | TPR  | TNR  |
|----------|----------------|-------------------|------|------|
| Logistic | 0.57           | 0.52              | 6%   | 99%  |
| Bagging  | 0.57           | 0.52              | 6%   | 99%  |
| RF       | 0.57           | 0.52              | 6%   | 99%  |
| LDA      | 0.57           | 0.52              | 6%   | 99%  |
| Boosting | 0.569          | 0.518             | 5%   | 99%  |
| RPART*   | 0.566          | 0.561             | 51%  | 61%  |
| SVM*     | 0.566          | 0.561             | 51%  | 61%  |
| NB       | 0.553          | 0.5               | >1%  | <99% |

Ultimately, using the test set, it was discovered that the driving factors in a maximum security facility assignment, were White Race/Ethnicity and Latest Admission Type. These results provided input to the problem statement yet should be interpreted with caution as the model results were unclear. Based on the data provided and domain expertise, the Boosting model was selected as the “best fit”, however, the True Positive and True Negative Rates for the Boosting model were highly variant, likely causing the model to have a high bias. 

Limitations of this study included, a low number of predictors, predictor similarity, and computational abilities. In the future researchers should consider the Boosting model but continue to test multiple models with an increased predictor size, while also using the maximum computing power available for efficiency. 

