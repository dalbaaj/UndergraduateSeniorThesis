import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


from sklearn.impute import KNNImputer
from sklearn.preprocessing import Binarizer
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

# need to encode the pregnancy column, if > 1 change value to 1
# this can be done using binarizer!
binarizer = Binarizer(threshold = 1)
binarizer.transform(np.array(data.loc[:,'Pregnancies']).reshape(-1,1))
#
data.hist()

# This tells us where there are 0 values in the data
# based on prior knowledge we can determine what 
# is realistic and what should not be zero
(data != 0).sum(0)


# imputation -- replace the 0's and outliers with np.nan

columns = data.columns[1:6]
for i in columns:
	data.loc[:,i].replace(0, np.nan, inplace = True)

# define KNN Imputer
imputer = KNNImputer(n_neighbors = 5, weights = 'uniform', metric = 'nan_euclidean')
X = data.drop(columns = ['Outcome'])
imputer.fit(X)
Xtrans = imputer.transform(X)

# need to use KNN to impute the missing values 
# determine what values of insulin are biologically impossible and treat those as missing and impute
# do not discard outliers