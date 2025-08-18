# CSE151A_Group_Project_SII25
The data and code for the CSE151A group project

# SETUP

Data for project is on [kaggle](https://www.kaggle.com/datasets/sherrytp/airline-delay-analysis). A kaggle account with API key is required. It must be downloaded and unzipped into Colab Notebook session.


# Submission 2 questions
**1. How many observations does your dataset have?**

 For simplicity, we will only look at airline delay info for the year 2018, which has exactly 7,213,446 observations.
 
**2. Describe all columns in your dataset their scales and data distributions. Describe the categorical and continuous variables in your dataset. Describe your target column and if you are using images plot some example classes of the images.**

 _FL_DATE_ : date. Distribution: flat distribution.

_OP_CARRIER_ : airline code (category). Distribution: some airlines show up more often.

_OP_CARRIER_FL_NUM_ : flight number (category). Distribution: some flight numbers show up the most.

_ORIGIN_ : origin airport code (category). Distribution: 90% unique airport codes.

_DEST_ : destination airport code (category). Distribution: 90% mostly unique.

_CRS_DEP_TIME_ : scheduled departure time (continuous). Distribution: most popular departure times are either morning or afternoon.

_DEP_TIME_ : actual departure time (continuous). Distribution: extremely inconsistent pattern.

_DEP_DELAY_ : minutes late/early at departure (continuous). Distribution: Most flights depart early.

_TAXI_OUT_ : minutes from gate pushback to takeoff (continuous). Distribution: Most flights have values 8-16 minutes.

_WHEELS_OFF_ : takeoff time (continuous). Distribution: extremely inconsistent pattern.

_WHEELS_ON_ : touchdown time (continuous). Distribution: extremely inconsistent pattern.

_TAXI_IN_ : minutes from touchdown to gate (continuous). Distribution: Minutes are mostly within 1-6

_CRS_ARR_TIME_ : scheduled arrival time (continuous). Distribution: extremely inconsistent pattern

_ARR_TIME_ : actual arrival time (continuous). Distribution: extremely inconsistent pattern

_ARR_DELAY_ : minutes late/early at the arrival gate (continuous). Distribution: Delays mostly range between ~50 minutes late to 1 hour early

_CANCELLED_ : 0/1 (category). Distribution: less than 2% are cancelled

_CANCELLATION_CODE_ : reason A/B/C/D (category and only for cancelled). Distribution: 98% empty values

_DIVERTED_ : 0/1 (category). Distribution: less than 1% flights diverted

_CRS_ELAPSED_TIME_ : planned gate-to-gate minutes (continuous). Distribution: Most values range roughly between 1 to 2 hours

_ACTUAL_ELAPSED_TIME_ : actual gate-to-gate minutes (continuous). Distribution: Most values also roughly range between 1 to 2 hours

_AIR_TIME_ : minutes in the air (continuous). Distribution: Most flights in the air range between 40 minutes to 1.5 hours

_DISTANCE_ : miles between airports (continuous). Distribution: most flights travel on average 800 miles

_CARRIER_DELAY_ : minutes attributed to airline (continuous). Distribution: Average time is 20 minutes

_WEATHER_DELAY_ : minutes due to weather (continuous). Distribution: Average time is 3.64 minutes

_NAS_DELAY_ : minutes due to airspace/ATC/volume (continuous). Distribution: Average time is 15.9 minutes

_SECURITY_DELAY_ : minutes due to security (continuous). Distribution: Average time is 0.09 minutes

_LATE_AIRCRAFT_DELAY_ : minutes due to late inbound plane (continuous). Distribution: Average time is 25.6 minutes

**3. Do you have missing and duplicate values in your dataset?**

 There is a significant number of missing data. This includes some of the feature columns we deem crucial to our target.

**4. Note: For image data you can still describe your data by the number of classes, size of images, are sizes uniform? Do they need to be cropped? normalized? etc.**

Some columns are categorical data, including date/time, that will require encoding. The target feature will have to be normalized.

**6. How will you preprocess your data? Handle data imbalance if needed. You should only explain (do not perform pre-processing as that is in MS3) this in your README.md file and link your Jupyter notebook to it. All code and  Jupyter notebooks have be uploaded to your repo.**

Some of the columns in the raw data will be dropped since they are irrelevant to the purpose of the model.
Missing data will be replaced with synthetic data computed from information available, if not possible to compute those values, they will be replaced with zeros. Categorical and date/time data will be encoded. Since this is a regression model, we do not anticipate significant issues with data imbalances, but we have enough data points to be able to drop rows for balancing purposes if necessary.

**7. You must also include in your Jupyter Notebook, a link for data download and environment setup requirements**

