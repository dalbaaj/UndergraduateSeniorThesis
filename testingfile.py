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
from sklearn.metrics import f1_score, auc, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve


# load data
data = pd.read_csv(r'C:\Users\danah\Documents\SeniorThesis\diabetes.csv')
# split data into X and y
X = data.drop(columns = ['Outcome'])
y = data['Outcome']

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

skf = StratifiedKFold(n_splits=10)

clf = Pipeline([
        ('scaler', scaler), 
        ('imputer', imputer),
        ('XGBoost', model)])


#defining various steps required for the genetic algorithm
def initialization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool) #creates bool array length of feature array
        chromosome[:int(0.3*n_feat)]=False # creates 2 false entries
        np.random.shuffle(chromosome) # shuffles array 
        population.append(chromosome)
    return population

def fitness_score(population, Xtr, ytr, Xts, yts):
    scores = []
    for chromosome in population:
        clf.fit(Xtr.iloc[:, chromosome], ytr)
        predictions = clf.predict(Xts.iloc[:, chromosome])
        scores.append(f1_score(yts, predictions)) # predictions must be 1d array
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
        scores, pop_after_fit = fitness_score(population, Xtr, ytr, Xts, yts)
        #print(scores[:2])
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population = mutation(pop_after_cross, r_mut)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo, best_score


def draw_cv_pr_curve(classifier, cv, X, y, title='PR Curve'):
    """
    Draw a Cross Validated PR Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series

    Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    """
    y_real = []
    y_proba = []

    i = 0
    for i, (train, test) in enumerate(cv.split(X, y)):
        # create testing and training sets
        Xtr = X.iloc[train]
        Xts = X.iloc[test]
        ytr = y.iloc[train]
        yts = y.iloc[test]

        # run genetic algorithm for these testing and training sets
        chromo, score = generations(n_pop=200, n_feat=8, n_parents = 10, r_mut=0.05, n_gen=20,
        Xtr=Xtr, Xts = Xts, ytr=ytr, yts = yts)
        probas_ = classifier.fit(Xtr.iloc[:,chromo[-1]], ytr).predict_proba(Xts.iloc[:, chromo[-1]])
        # Compute ROC curve and area the curve
        precision, recall, _ = precision_recall_curve(yts, probas_[:, 1])

        # Plotting each individual PR Curve
        plt.plot(recall, precision, lw=1, alpha=0.3,
                 label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(yts, probas_[:, 1])))

        y_real.append(yts)
        y_proba.append(probas_[:, 1])


    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    plt.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def draw_cv_norm_conf_matrix(classifier, cv, X, y):
    """
    Draw a Cross Validated PR Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series

    Largely taken from: https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn
    """
    # empty list for confusion matrix 
    conf_matrix_list_of_arrays = []
    for i, (train, test) in enumerate(cv.split(X, y)):
        # create testing and training sets
        Xtr = X.iloc[train]
        Xts = X.iloc[test]
        ytr = y.iloc[train]
        yts = y.iloc[test]

        # run genetic algorithm for these testing and training sets
        chromo, score = generations(n_pop=200, n_feat=8, n_parents = 10, r_mut=0.05, n_gen=20,
        Xtr=Xtr, Xts = Xts, ytr=ytr, yts = yts)
        
        # get information for confusion matrix
        conf_matrix = confusion_matrix(yts, classifier.predict(Xts.iloc[:,chromo[-1]]), labels = classifier.classes_, normalize = 'true') # normalized confusion matrix -> eases interpretation
        conf_matrix_list_of_arrays.append(conf_matrix)
    
    mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=mean_of_conf_matrix_arrays, display_labels=classifier.classes_) 
    disp.plot()


def draw_cv_roc_curve(classifier, cv, X, y, title='ROC Curve'):
    """
    Draw a Cross Validated ROC Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series
    Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cv.split(X, y)):
        # create testing and training sets
        Xtr = X.iloc[train]
        Xts = X.iloc[test]
        ytr = y.iloc[train]
        yts = y.iloc[test]

        # run genetic algorithm for these testing and training sets
        chromo, score = generations(n_pop=200, n_feat=8, n_parents = 10, r_mut=0.05, n_gen=20,
        Xtr=Xtr, Xts = Xts, ytr=ytr, yts = yts)
        probas_ = classifier.fit(Xtr.iloc[:,chromo[-1]], ytr).predict_proba(Xts.iloc[:, chromo[-1]])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(yts, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))


    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

draw_cv_pr_curve(clf, skf, X, y)
#draw_cv_norm_conf_matrix(clf, skf, X, y)
draw_cv_roc_curve(clf, skf, X, y)
