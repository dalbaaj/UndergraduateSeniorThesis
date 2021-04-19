
# First XGBoost model for Pima Indians dataset
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
data = pd.read_csv(r'C:\Users\danah\Documents\SeniorThesis\diabetes.csv')
# split data into X and y
X = data.drop(columns = ["Outcome"])
Y = data['Outcome'].to_numpy()
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Preprocess the data 
# imputation -- replace these 0's with the median of the row 
X_train.Glucose.replace(0, np.median(X_train.Glucose), inplace = True)
X_train.Insulin.replace(0, np.median(X_train.Insulin), inplace = True)
X_train.BloodPressure.replace(0, np.median(X_train.BloodPressure), inplace = True)
X_train.SkinThickness.replace(0, np.median(X_train.SkinThickness), inplace = True)
X_train.BMI.replace(0, np.median(X_train.BMI), inplace = True)
X_test.Glucose.replace(0, np.median(X_test.Glucose), inplace = True)
X_test.Insulin.replace(0, np.median(X_test.Insulin), inplace = True)
X_test.BloodPressure.replace(0, np.median(X_test.BloodPressure), inplace = True)
X_test.SkinThickness.replace(0, np.median(X_test.SkinThickness), inplace = True)
X_test.BMI.replace(0, np.median(X_test.BMI), inplace = True)

# need to encode the pregnancy column, if > 1 change value to 1
X_train.loc[X_train['Pregnancies'] != 0, 'Pregnancies'] = 1
X_test.loc[X_test['Pregnancies'] != 0, 'Pregnancies'] = 1

# fit model to training data
model = XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc', use_label_encoder=False)
model.fit(X_train, y_train)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# note everything in the dataset is a numerical value, just like the example below

#defining various steps required for the genetic algorithm
def initialization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population):
    scores = []
    for chromosome in population:
        model.fit(X_train.iloc[:,chromosome],y_train)
        predictions = model.predict(X_test.iloc[:,chromosome])
        scores.append(accuracy_score(y_test,predictions))
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

def generations(n_pop, n_feat, n_parents, r_mut, n_gen, X_train, X_test, y_train, y_test):
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
chromo, score = generations(n_pop=200, n_feat=8, n_parents = 10, r_mut=0.1,n_gen=38,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
model.fit(X_train.iloc[:,chromo[-1]],y_train)
predictions = model.predict(X_test.iloc[:,chromo[-1]])
print("Accuracy score after genetic algorithm is= "+str(accuracy_score(y_test,predictions)))




