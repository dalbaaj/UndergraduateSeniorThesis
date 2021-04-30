import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
%matplotlib inline

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.compose import ColumnTransformer
# from sklearn.compose import make_column_selector

from sklearn.preprocessing import Binarizer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# load data
data = pd.read_csv(r'C:\Users\danah\Documents\SeniorThesis\diabetes.csv')
# split data into X and y
data.Pregnancies = np.where(data.Pregnancies != 0, 1, data.Pregnancies)
X = data.drop(columns = ['Outcome'])
y = data['Outcome'].to_numpy()


missing_vals_ix = X.columns[1:6]

for i in missing_vals_ix:
    X.loc[:, i].replace(0, np.nan, inplace = True)

imputer = KNNImputer(weights='distance')
model = xgb.XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc', use_label_encoder=False, learning_rate = 0.1, n_estimators = 10, max_depth = 5, alpha = 10)


clf = Pipeline(steps=[('imputer', imputer), ('XGBoost', model)])


# split data into train and test sets
seed = 7
test_size = 0.3
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed)
# need to do cross validation up here, clf should represent the pipeline
# need to build the cv into the pipeline
# so the split of the training and testing happens multiple times!

scores = cross_val_score(clf, X, y, cv = 5)


# make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
print("Accuracy score before genetic algorithm: %.2f%%" % (np.mean(scores) * 100.0))

# note everything in the dataset is a numerical value, just like the example below

#defining various steps required for the genetic algorithm
def initialization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool) #creates bool array length of feature array
        chromosome[:int(0.3*n_feat)]=False # creates 2 false entries
        np.random.shuffle(chromosome) # shuffles array 
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    for chromosome in population:
        cv_score = cross_val_score(clf, X.iloc[:,chromosome], y, cv = 10)
        scores.append(np.mean(cv_score))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])


def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

# may need to manipulate this function, don't know how the child is being created
def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

def mutation(pop_after_cross, r_mut):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < r_mut:
                chromosome[j]= not chromosome[j]
        population_nextgen.append(chromosome)
    #print(population_nextgen)
    return population_nextgen

def generations(n_pop, n_feat, n_parents, r_mut, n_gen, X, y):
    population = initialization_of_population(n_pop,n_feat)
    # keep track of best solution
    best_chromo = []
    best_score = []
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population)
        #print(scores[:2])
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population = mutation(pop_after_cross, r_mut)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo, best_score

#increased n_pop, n_parents, and r_mut led to decreased AUC
# increased n_pop, n_parents and decreased r_mut led to increased AUC!
chromo, score = generations(n_pop=400, n_feat=8, n_parents = 50, r_mut=0.001, n_gen=38, X=X, y=y)
new_scores = cross_val_score(clf, X.iloc[:,chromo[-1]], y, cv = 10)
print("Accuracy score after genetic algorithm is: %.2f%%" % (np.mean(new_scores)*100.00))


# try to do repeated k-fold cv
# data_dmatrix = xgb.DMatrix(data=X,label=y)
# params = {"objective":"binary:logistic",'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}

