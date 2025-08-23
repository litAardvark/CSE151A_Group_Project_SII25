# Meeting Notes 08/23/25
## Objective 
- Predict ARR_DELAY (# minutes plane is delayed on arrival) using Support Vector Regression (SVR)
## General 
- Jason will run all of our code through SDSC, use small subset of data when working locally
- We are allowed to use imports for everything !!! 
## Data Preprocessing
- Encoding time data cyclically as floats using sin() 
- 1-hot encoding categoricals for use in future submission 
- Fill in missing categoricals with mode
- Fill in missing numerical data with mean
- Scaling numerical data using z-scoring
- Subset data to size 500,000 rows
## Training Model 
- Will train on a subset of size  including only numerical data for Milestone 3 since we are on a time crunch

## Division of Labor 
- Jason: Running code on SDSC
- DQ:Preprocessing (z-scoring, filling in missing numericals, encoding time data) 
- Rosario:Preprocessing  (Filling in categorical stuff) 
- Sam: Writing
  


