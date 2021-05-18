import numpy as np
import pandas as pd
import random as random
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
#%matplotlib inline

# from sklearn.model_selection import train_test_split


from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, plot_roc_curve, confusion_matrix, ConfusionMatrixDisplay


# load data
data = pd.read_csv(r'C:\Users\danah\Documents\SeniorThesis\diabetes.csv')
# split data into X and y
X = data.drop(columns = ['Outcome'])
y = data['Outcome'].to_numpy()

X.Pregnancies = np.where(X.Pregnancies != 0, 1, X.Pregnancies)
missing_cols = tuple([X.columns[i] for i in range(1,7)])
X.loc[:, missing_cols].replace(0, np.nan, inplace = True)

# split data into train and test sets
seed = 7
test_size = 0.3
scaler = RobustScaler(quantile_range=(1.0, 99.0), with_centering=False)
imputer = KNNImputer(weights='distance')

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

skf = StratifiedKFold(n_splits=5)

clf = Pipeline([
        ('scaler', scaler), 
        ('imputer', imputer),
        ('XGBoost', model)])

cv_results = cross_validate(clf, X, y, cv = skf, scoring = 'roc_auc')
print("Accuracy score before genetic algorithm: %.2f%%" % (np.mean(cv_results['test_score']) * 100.0))

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
        clf.fit(Xtr.iloc[:, chromosome], ytr)
        predictions = clf.predict_proba(Xts.iloc[:, chromosome])
        scores.append(roc_auc_score(yts, predictions[:,1])) # predictions must be 1d array
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

def generations(n_pop, n_feat, n_parents, r_mut, n_gen, Xtr, Xts, ytr, yts):
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

# empty values for auc curves
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

# empty list for confusion matrix 
conf_matrix_list_of_arrays = []

fig, ax = plt.subplots()
for i, (train, test) in enumerate(skf.split(X, y)):
    # create testing and training sets
    Xtr = X.iloc[train]
    Xts = X.iloc[test]
    ytr = y[train]
    yts = y[test]

    # run genetic algorithm for these testing and training sets
    chromo, score = generations(n_pop=200, n_feat=8, n_parents = 10, r_mut=0.01, n_gen=40,
        Xtr=Xtr, Xts = Xts, ytr=ytr, yts = yts)
    
    # fit the model to result of genetic algorithm
    clf.fit(Xtr.iloc[:, chromo[-1]], ytr)

    # get information for auc curve
    viz = plot_roc_curve(clf, Xts.iloc[:, chromo[-1]], yts, name = 'ROC fold {}'.format(i))
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    # get information for confusion matrix
    conf_matrix = confusion_matrix(yts, clf.predict(Xts.iloc[:, chromo[-1]]), labels = clf.classes_)
    conf_matrix_list_of_arrays.append(conf_matrix)


ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver Operating Characteristic for Proposed Model")
ax.legend(loc="lower right")
plt.show()

mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
disp = ConfusionMatrixDisplay(confusion_matrix=mean_of_conf_matrix_arrays, display_labels=clf.classes_)
disp.plot()
