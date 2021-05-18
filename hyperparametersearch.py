import numpy as np
import pandas as pd
import random as random
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
#%matplotlib inline

# from sklearn.preprocessing import FunctionTransformer
# from sklearn.compose import ColumnTransformer
# from sklearn.compose import make_column_selector

from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import classification_report, confusion_matrix

# load data
data = pd.read_csv(r'C:\Users\danah\Documents\SeniorThesis\diabetes.csv')
# split data into X and y
X = data.drop(columns = ['Outcome'])
y = data['Outcome'].to_numpy()

# Binarize the pregnancies
X.Pregnancies = np.where(X.Pregnancies != 0, 1, X.Pregnancies)


# need to scale data before KNN

scaler = RobustScaler(quantile_range=(5.0, 95.0))

missing_cols = tuple([X.columns[i] for i in range(1,7)])
zero_replacement(missing_cols)

imputer = KNNImputer(weights='distance')
# need to tune XGBoost hyperparameters, can use GridSearchCV
model = XGBClassifier( 
learning_rate =0.2, 
n_estimators=1000, 
max_depth=2,
min_child_weight=2, 
gamma=0.2, 
subsample=0.6, 
colsample_bytree=0.55,
reg_alpha = 1e-5,
objective= 'binary:logistic', 
eval_metric = 'auc', 
nthread=4, 
scale_pos_weight=1, 
seed=27, 
use_label_encoder = False)
clf = Pipeline([
        ('scaler', scaler), 
        ('imputer', imputer),
        ('XGBoost', xgb3)])


Xtr, Xts, ytr, yts = train_test_split(X, y, test_size = 0.33)

# base line model
clf.fit(Xtr, ytr)
prediction = clf.predict(Xts)
print(confusion_matrix(yts,prediction))
print(classification_report(yts,prediction))
acc1 = accuracy_score(yts,prediction)
print("Accuracy score before genetic algorithm: %.2f%%" % (acc1 * 100.0))


# Step 1: Fix learning rate and number of estimators for tuning tree-based parameters

def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("Model Report")
    print('The optimal # of estimators is:', alg.get_params()['n_estimators'])
    print("Accuracy : %.4g" % accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(dtrain[target], dtrain_predprob))


                    

data = pd.read_csv(r'C:\Users\danah\Documents\SeniorThesis\diabetes.csv')
target = 'Outcome'
#Choose all predictors
predictors = [X for X in data.columns if X != target]
xgb1 = XGBClassifier(
 learning_rate =0.2,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27, 
 eval_metric='auc'
 use_label_encoder=False)
modelfit(xgb1, data, predictors)

# Step 2: Tune max_depth and min_child_weight

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=24, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', eval_metric = 'auc', nthread=4, scale_pos_weight=1, seed=27, use_label_encoder = False), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)
gsearch1.fit(data[predictors],data[target])
print(gsearch1.best_params_, '\n', gsearch1.best_score_)


# fine tuning of the above parameter search 
# reduce interval to one above and one below, step size of 1
param_test2 = {
 'max_depth':[1,2,3,4],
 'min_child_weight':[1,2,3,4]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=24, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic', 
 eval_metric = 'auc', nthread=4, scale_pos_weight=1, seed=27, use_label_encoder = False), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4, cv=5)
gsearch2.fit(data[predictors],data[target])
print('Best Parameter values:', gsearch2.best_params_, '\nAUC:', gsearch2.best_score_)

# improvement with lower vals at 2,2


# Step 3: Tune Gamma
param_test3 = {
 'gamma': [i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=24, max_depth=2,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', 
 eval_metric = 'auc', nthread=4, scale_pos_weight=1, seed=27, use_label_encoder = False), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4, cv=5)
gsearch3.fit(data[predictors],data[target])
print('Best Parameter values:', gsearch3.best_params_, '\nAUC:', gsearch3.best_score_)


# Recalibrate boosting rounds for these new optimal parameter values
xgb2 = XGBClassifier( 
learning_rate =0.2, 
n_estimators=1000, 
max_depth=2,
min_child_weight=2, 
gamma=0.2, 
subsample=0.8, 
colsample_bytree=0.8,
objective= 'binary:logistic', 
eval_metric = 'auc', 
nthread=4, 
scale_pos_weight=1, 
seed=27, 
use_label_encoder = False)
modelfit(xgb2, data, predictors)


# Step 4: Tune subsample and colsample_bytree
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=23, max_depth=2,
 min_child_weight=2, gamma=0.2, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', 
 eval_metric = 'auc', nthread=4, scale_pos_weight=1, seed=27, use_label_encoder = False), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4, cv=5)
gsearch4.fit(data[predictors],data[target])
print('Best Parameter values:', gsearch4.best_params_, '\nAUC:', gsearch4.best_score_)

# Fine tune these results
param_test5 = {
 'subsample':[i/100.0 for i in range(55,70,5)],
 'colsample_bytree':[i/100.0 for i in range(55,70,5)]
}

gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=23, max_depth=2,
 min_child_weight=2, gamma=0.2, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', 
 eval_metric = 'auc', nthread=4, scale_pos_weight=1, seed=27, use_label_encoder = False), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4, cv=5)
gsearch5.fit(data[predictors],data[target])
print('Best Parameter values:', gsearch5.best_params_, '\nAUC:', gsearch5.best_score_)

# both are now 0.65

# Step 5: Tuning Regularization Parameters
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.2, n_estimators=23, max_depth=2,
 min_child_weight=2, gamma=0.2, subsample=0.6, colsample_bytree=0.55, objective= 'binary:logistic', 
 eval_metric = 'auc', nthread=4, scale_pos_weight=1, seed=27, use_label_encoder = False), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4, cv=5)
gsearch6.fit(data[predictors],data[target])
print('Best Parameter values:', gsearch6.best_params_, '\nAUC:', gsearch6.best_score_)


#check impact of Regularization and other parameters on model
# make this our model!
xgb3 = XGBClassifier( 
learning_rate =0.2, 
n_estimators=1000, 
max_depth=2,
min_child_weight=2, 
gamma=0.2, 
subsample=0.6, 
colsample_bytree=0.55,
reg_alpha = 1e-5,
objective= 'binary:logistic', 
eval_metric = 'auc', 
nthread=4, 
scale_pos_weight=1, 
seed=27, 
use_label_encoder = False)
modelfit(xgb3, data, predictors)



# lowering the learning rate had a negative impact - redo with a higher learning rate (try 0.2) - results are above


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
        cv_score = cross_val_score(clf, X.iloc[:,chromosome], y, cv = skf)
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

# increased n_pop, n_parents, and r_mut led to decreased AUC
# increased n_pop, n_parents and decreased r_mut led to increased AUC!
n_pop = [100, 200, 300, 400, 500] 
n_parents = [10, 20, 30, 40, 50]
r_mut = [0.05, 0.04, 0.03, 0.02, 0.01]
n_gen = [35, 45, 55, 65, 75]


# search for best pop value
for i in range(len(n_pop)):
    chromo, score = generations(n_pop=n_pop[i], n_feat=8, n_parents = 10, r_mut=0.01, n_gen=40, X=X, y=y)
    new_scores = cross_val_score(clf, X.iloc[:,chromo[-1]], y, cv = skf)
    print(str(n_pop[i]) + " accuracy score is: %.2f%%" % (np.mean(new_scores)*100.00))

# search for best parent value
for i in range(len(n_parents)):
    chromo, score = generations(n_pop=300, n_feat=8, n_parents = n_parents[i], r_mut=0.01, n_gen=40, X=X, y=y)
    new_scores = cross_val_score(clf, X.iloc[:,chromo[-1]], y, cv = skf)
    print(str(n_parents[i]) + " accuracy score is: %.2f%%" % (np.mean(new_scores)*100.00))

# search for best r_mut value
for i in range(len(r_mut)):
    chromo, score = generations(n_pop=300, n_feat=8, n_parents = , r_mut=r_mut[i], n_gen=40, X=X, y=y)
    new_scores = cross_val_score(clf, X.iloc[:,chromo[-1]], y, cv = skf)
    print(str(r_mut[i]) + " accuracy score is: %.2f%%" % (np.mean(new_scores)*100.00))

# verify the results of this model is going up
chromo, score = generations(n_pop=300, n_feat=8, n_parents = , r_mut=, n_gen=40, X=X, y=y)
new_scores = cross_val_score(clf, X.iloc[:,chromo[-1]], y, cv = skf)
print("Accuracy score after genetic algorithm is: %.2f%%" % (np.mean(new_scores)*100.00))

# see if number of generations will have an impact on score
for i in range(len(n_gen)):
    chromo, score = generations(n_pop=300, n_feat=8, n_parents = , r_mut=, n_gen=n_gen[i], X=X, y=y)
    new_scores = cross_val_score(clf, X.iloc[:,chromo[-1]], y, cv = skf)
    print(str(n_gen[i]) + " accuracy score is: %.2f%%" % (np.mean(new_scores)*100.00))