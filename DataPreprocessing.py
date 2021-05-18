import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


from sklearn.impute import KNNImputer
from sklearn.preprocessing import Binarizer, RobustScaler
data = pd.read_csv(r'C:\Users\danah\Documents\SeniorThesis\diabetes.csv')
data.head()

# statistical summary and distribution

print('The statistical summary of the data is: \n', data.describe())
print('The data distribution is: ', data.groupby('Outcome').size())

# the visualization of the data -- gather the understanding of the outliers and realistic values
# need to change the size of the figures
colors = ['green', 'orange', 'tan', 'magenta', 'red', 'deeppink', 'purple', 'skyblue', 'royalblue']
col = data.columns
for i in range(len(col)): 
    plt.hist(data.loc[:,col[i]], color=colors[i])
    plt.title(str(col[i]))
    plt.show()

# This tells us where there are 0 values in the data
# Prior knowledge determines what is realistic and what should not be zero
(data != 0).sum(0)


# acts like Binarizer -> set all values != 0 as 1
data.Pregnancies = np.where(data.Pregnancies != 0, 1, data.Pregnancies)

# imputation -- replace the 0's with np.nan
columns = data.columns[1:6]
for i in columns:
	data.loc[:,i].replace(0, np.nan, inplace = True)

# must scale data before imputation
scaler = RobustScaler(quantile_range=(1.0, 99.0), with_centering = False)
X = data.drop(columns = ['Outcome'])
scaler.fit(X)
scaledX = scaler.transform(X)

# define KNN Imputer
imputer = KNNImputer(n_neighbors = 5, weights = 'distance', metric = 'nan_euclidean')
imputer.fit(scaledX)
Xtrans = imputer.transform(scaledX)

# visualize results
for j in range(Xtrans.shape[1]):
    plt.hist(Xtrans[:,j], color= colors[j])
    plt.title(str(col[j]) + ' After Imputation')
    plt.show()


