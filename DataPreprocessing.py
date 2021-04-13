import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv(r'C:\Users\danah\Documents\SeniorThesis\diabetes.csv')
data.head()

# statistical summary and distribution
print('The statistical summary of the data is: \n', data.describe())
print('The data distribution is: ', data.groupby('Outcome').size())

# the visualization of the data -- gather the understanding of the outliers and realistic values
# need to change the size of the figures
data.plot(kind='box', subplots = True, layout = (3,3), sharex = False, sharey = False)
data.hist()
plt.show()

# This tells us where there are 0 values in the data
# based on prior knowledge we can determine what 
# is realistic and what should not be zero
(data != 0).sum(0)

# the number of zero values that should not exist in the dataset are located here
data[data.Glucose == 0].shape[0]
data[data.BloodPressure == 0].shape[0]
data[data.Insulin == 0].shape[0]
data[data.BMI == 0].shape[0]
data[data.SkinThickness == 0].shape[0]

# imputation -- replace these 0's with the median of the row 
data.Glucose.replace(0, np.median(data.Glucose), inplace = True)
data.Insulin.replace(0, np.median(data.Insulin), inplace = True)
data.BloodPressure.replace(0, np.median(data.BloodPressure), inplace = True)
data.SkinThickness.replace(0, np.median(data.SkinThickness), inplace = True)
data.BMI.replace(0, np.median(data.BMI), inplace = True)

# replot the histograms to show the new distributions
(data != 0).sum(0)


# will need to go back and do this all over again with the scikit learn package 
# must first separate into training and testing set in order to prevent any bias and other issues