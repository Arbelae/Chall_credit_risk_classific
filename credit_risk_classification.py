#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Import the modules
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


# ---

# ## Split the Data into Training and Testing Sets

# ### Step 1: Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.

# In[11]:


# Read the CSV file from the Resources folder into a Pandas DataFrame
df = pd.read_csv("lending_data.csv")

# Review the DataFrame
df.head()


# ### Step 2: Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.

# In[12]:


# Separate the data into labels and features
# Separate the y variable, the labels
Info = df["loan_status"]

# Separate the X variable, the features
data = df.drop("loan_status", axis=1)
feature_names = data.columns


# In[13]:


# Review the y variable Series
Info.head()


# In[14]:


# Review the X variable DataFrame
data.head()


# ### Step 3: Split the data into training and testing datasets by using `train_test_split`.

# In[22]:


# Import the train_test_learn module
from sklearn.model_selection import train_test_split

# Split the data using train_test_split
# Assign a random_state of 1 to the function

X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    Info, 
                                                    random_state=1, 
                                                    stratify=Info)


# ## Create a Logistic Regression Model with the Original Data

# ###  Step 1: Fit a logistic regression model by using the training data (`X_train` and `y_train`).

# In[23]:


# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
classifier = LogisticRegression(solver='lbfgs', random_state=1)

# Fit the model using training data
classifier.fit(X_train, y_train)


# ### Step 2: Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

# In[25]:


# Make a prediction using the testing data
predictions = classifier.predict(X_test)
pd.DataFrame({"Prediction": predictions, "Actual": y_test})


# ### Step 3: Evaluate the model’s performance by doing the following:
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[32]:


# Generate a confusion matrix for the model
confusion_matrix(y_test, predictions)


# In[33]:


# Print the classification report for the model
print (classification_report(y_test, predictions))


# ### Step 4: Answer the following question.

# **Question:** How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
# 
# **Answer:** The logistic regression model seems to be performing quite well in predicting both healthy (0) and high-risk (1) loans based on the accuracy and precision values.
# 
# An accuracy of 0.99 indicates that the model is able to correctly classify 99% of the instances in the dataset. This is generally a very high accuracy rate and suggests that the model is effective in making correct predictions for both healthy and high-risk loans.
# 
# Additionally, a precision of 0.87 for the high-risk (1) label indicates that when the model predicts a loan as high-risk, it is correct approximately 87% of the time. This is a good precision value, as it suggests that the model is reliable in identifying high-risk loans and does not falsely classify too many healthy loans as high-risk.
# 
# Overall, based on these metrics, it appears that the logistic regression model is highly effective in predicting both healthy and high-risk loans.
