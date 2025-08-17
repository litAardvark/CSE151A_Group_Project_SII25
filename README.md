# CSE151A_Group_Project_SII25
The data and code for the CSE151A group project

# SETUP

Data for project is on [kaggle](https://www.kaggle.com/datasets/sherrytp/airline-delay-analysis). A kaggle account with API key is required. It must be downloaded and unzipped into Colab Notebook session.


# Submission 2 questions
**1. How many observations does your dataset have?**

 /
 
**2. Describe all columns in your dataset their scales and data distributions. Describe the categorical and continuous variables in your dataset. Describe your target column and if you are using images plot some example classes of the images.**

 /

**3. Do you have missing and duplicate values in your dataset?**

 There is a significant number of missing data. This includes some of the columns crucial to computing our target feature, total delayed time.

**4. Note: For image data you can still describe your data by the number of classes, size of images, are sizes uniform? Do they need to be cropped? normalized? etc.**

Some columns are categorical data, including date and time, that will require encoding. The yet uncomputed total delay time will have to be normalized.

**6. How will you preprocess your data? Handle data imbalance if needed. You should only explain (do not perform pre-processing as that is in MS3) this in your README.md file and link your Jupyter notebook to it. All code and  Jupyter notebooks have be uploaded to your repo.**

Most of the columns in the raw data will be dropped. Missing data will be replaced with synthetic data computed from information available, if not possible to compute those values, they will be replaced with zeros. Categorical and date/time data will be encoded. Our target feature is not explicit in the dataset so it will have to be computed. Since this is a regression model, we do not anticipate significant issues with data imbalances.

**7. You must also include in your Jupyter Notebook, a link for data download and environment setup requirements**

