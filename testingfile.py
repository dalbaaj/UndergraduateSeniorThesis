import numpy as np
import pandas as pd
import random as random
import matplotlib.pyplot as plt
from sklearn.metrics._plot.precision_recall_curve import plot_precision_recall_curve
from sklearn.metrics._plot.roc_curve import plot_roc_curve
import xgboost as xgb
from xgboost import XGBClassifier
#%matplotlib inline
from numpy import interp

from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, auc, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve


# load data
data = pd.read_csv(r'C:\Users\danah\Documents\SeniorThesis\diabetes.csv')
# split data into X and y
X = data.drop(columns = ['Outcome'])
y = data['Outcome']

# from data preprocessing, binarize pregnancy column
X.Pregnancies = np.where(X.Pregnancies != 0, 1, X.Pregnancies)

# replace disguised missing values with np.nan for KNN imputation
missing_cols = tuple([X.columns[i] for i in range(1,7)])
X.loc[:, missing_cols].replace(0, np.nan, inplace = True)

# remove extreme outliers & scale data to improve results of KNN imputation
scaler = RobustScaler(quantile_range=(1.0, 99.0), with_centering=False)
imputer = KNNImputer(weights='distance')

# values pulled from hyperparameter testing
model = XGBClassifier( 
learning_rate =0.2, 
n_estimators=25, # changed this to 25 to see if speed improves
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

# performs KFold while maintaining the proper ratio for class imbalance
skf = StratifiedKFold(n_splits=10)

clf = Pipeline([
        ('scaler', scaler), 
        ('imputer', imputer),
        ('XGBoost', model)])

# get baseline cross validated metrics
def baseline_metrics(classifier, cv, X, y, title1, title2):
    """
    Draw Baseline Cross Validated Metrics.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series
    Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating lists for PR Curve
    y_real = []
    y_proba = []

    # Creating lists for ROC Curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    conf_matrix_list_of_arrays = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for i, (train, test) in enumerate(cv.split(X, y)):
        # create testing and training sets
        Xtr = X.iloc[train]
        Xts = X.iloc[test]
        ytr = y.iloc[train]
        yts = y.iloc[test]

        # fit classifier to training data with optimal feature set
        classifier.fit(Xtr, ytr)

        # find predictions for optimal feature set
        yhat = classifier.predict(Xts)

        # get confusion matrix and store
        conf_matrix = confusion_matrix(yts, yhat, labels = classifier.classes_, normalize = 'true') # normalized confusion matrix -> eases interpretation
        conf_matrix_list_of_arrays.append(conf_matrix)

        # find probabilities of testing data w/ optimal feature set
        probas_ = classifier.predict_proba(Xts)
        # Compute pr curve and area under the curve
        precision, recall, _ = precision_recall_curve(yts, probas_[:, 1])

        # Plotting each individual PR Curve
        ax1.plot(recall, precision, lw=1, alpha=0.3,
                 label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(yts, probas_[:, 1])))
        
        y_real.append(yts)
        y_proba.append(probas_[:, 1])

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(yts, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax2.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    # complete PR AUC figure
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    ax1.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.4f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    ax1.xaxis.set(ticks = [-0.05, 1.05])
    ax1.yaxis.set(ticks = [-0.05, 1.05])
    ax1.set(title = title1, xlabel = 'Recall', ylabel = 'Precision')
    ax1.legend(loc="upper right", bbox_to_anchor=(1.7, 1))

    # complete ROC AUC figure
    ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax2.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax2.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    ax2.xaxis.set(ticks = [-0.05, 1.05])
    ax2.yaxis.set(ticks =[-0.05, 1.05])
    ax2.set(title = title2, xlabel='False Positive Rate', ylabel = 'True Positive Rate')
    ax2.legend(loc="lower right", bbox_to_anchor=(1.7, 0))
    
    fig.subplots_adjust(wspace = 1) #Adjust the spacing between subplots
    plt.show()

    # complete Confusion Matrix Display
    mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=mean_of_conf_matrix_arrays, display_labels=classifier.classes_) 
    disp.plot()

baseline_metrics(clf, skf, X, y, title1='Baseline Model PR Curve', title2='Baseline Model ROC Curve')


#defining various steps required for the genetic algorithm
def initialization_of_population(size,n_feat):
    """
    Create initial random population, can be of any dtype. 
    Keyword args:
        size: Integer length of the initial population
        n_features: Integer amount of features in the dataset
    """
    # note to self: consider randomizing the 0.3 so the number of features that are false changes?
    population = []
    for i in range(size):
        n_false = np.random.rand(1) # generate % of features to make false
        chromosome = np.ones(n_feat, dtype=np.bool) #creates bool array length of feature array
        chromosome[:int(n_false*n_feat)]=False # creates false entries
        np.random.shuffle(chromosome) # shuffles array to create individual
        population.append(chromosome) # adds individual to population
    return population

def fitness_score(population, Xtr, ytr, Xts, yts):
    """
    Determine fitness scores of each individual in the population using classifier accuracy. 
    Keyword args:
        population: multi-dimensional array 
        Xtr: Training Feature Pandas DataFrame 
        ytr: Training Response Pandas Series Object
        Xts: Testing Feature Pandas DataFrame
        yts: Testing Response Pandas Series Object
    """
    scores = []
    for chromosome in population:
        clf.fit(Xtr.iloc[:, chromosome], ytr) 
        predictions = clf.predict(Xts.iloc[:, chromosome])
        scores.append(f1_score(yts, predictions)) # predictions must be 1d array
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])


def selection(pop_after_fit,n_parents):
    """
    Select n individuals for reproduction based on fitness scores. 
    Keyword args:
        pop_after_fit: multi-dimensional array of individuals sorted in descending order of fitness scores
        n_parents: number of parents desired
    """
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    """
    Create offspring to add to population from the selected individuals.  
    Keyword args:
        pop_after_sel: multi-dimensional array representing individuals selected as parents
    """
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        parent1 = pop_after_sel[i] 
        parent2 = pop_after_sel[(i+1)%len(pop_after_sel)]
        ix = np.random.randint(1, 7) # randomly select the index the parents use for crossover
        child = np.concatenate((parent1[:ix],parent2[ix:]))
        population_nextgen.append(child)
    return population_nextgen

def mutation(pop_after_cross, r_mut):
    """
    Define mutations in the population's offspring.   
    Keyword args:
        pop_after_cross: multi-dimensional array containing offspring
        r_mut: float value representing rate of mutation
    """
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < r_mut:
                chromosome[j] = not chromosome[j]
        population_nextgen.append(chromosome)
    return population_nextgen

def generations(n_pop, n_feat, n_parents, r_mut, n_gen, Xtr, Xts, ytr, yts):
    """
    Genetic Algorithm implementation for multiple generations to find optimal feature set. 
    Keyword args:
        n_pop: Size of the initial population
        n_feat: number of features in the dataset
        n_parents: desired number of parents
        r_mut: rate of mutation
        n_gen: number of generations expected
        Xtr: Training Feature Pandas DataFrame 
        Xts: Testing Feature Pandas DataFrame
        ytr: Training Response Pandas Series Object
        yts: Testing Response Pandas Series Object
    """
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

# heavy class imbalance calls for precision recall curve
def draw_cv_pr_metrics(classifier, cv, X, y, title):
    """
    Draw a Cross Validated PR Curve and Confusion Metrics.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series

    Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    and 
    https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn
    """
    y_real = []
    y_proba = []

    conf_matrix_list_of_arrays = []
    opt_chromos = []

    for i, (train, test) in enumerate(cv.split(X, y)):
        # create testing and training sets
        Xtr = X.iloc[train]
        Xts = X.iloc[test]
        ytr = y.iloc[train]
        yts = y.iloc[test]

        # run genetic algorithm for these testing and training sets
        chromo, score = generations(n_pop=200, n_feat=8, n_parents = 10, r_mut=0.03, n_gen=30,
        Xtr=Xtr, Xts = Xts, ytr=ytr, yts = yts)

        # storing the optimal feature sets
        opt_chromos.append(chromo[-1])

        # fit classifier to training data with optimal feature set
        classifier.fit(Xtr.iloc[:,chromo[-1]], ytr)

        # find predictions for optimal feature set
        yhat = classifier.predict(Xts.iloc[:, chromo[-1]])

        # get confusion matrix and store
        conf_matrix = confusion_matrix(yts, yhat, labels = classifier.classes_, normalize = 'true') # normalized confusion matrix -> eases interpretation
        conf_matrix_list_of_arrays.append(conf_matrix)

        # find probabilities of testing data w/ optimal feature set
        probas_ = classifier.predict_proba(Xts.iloc[:, chromo[-1]])
        # Compute pr curve and area under the curve
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
             label=r'Precision-Recall (AUC = %0.4f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0))
    plt.show()

    mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=mean_of_conf_matrix_arrays, display_labels=classifier.classes_) 
    disp.plot()

    
    print("Optimal features for pr metrics:", opt_chromos)


# necessary to compare the performance of this model to model's in previous work
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
    opt_chromos = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig = plt.figure()

    for i, (train, test) in enumerate(cv.split(X, y)):
        # create testing and training sets
        Xtr = X.iloc[train]
        Xts = X.iloc[test]
        ytr = y.iloc[train]
        yts = y.iloc[test]

        # run genetic algorithm for these testing and training sets
        chromo, score = generations(n_pop=200, n_feat=8, n_parents = 10, r_mut=0.03, n_gen=30,
        Xtr=Xtr, Xts = Xts, ytr=ytr, yts = yts)
        # store optimal feature set
        opt_chromos.append(chromo[-1])

        # fit classifier on training set w/ optimal features selected by GA
        classifier.fit(Xtr.iloc[:,chromo[-1]], ytr)
        # prediction probabilities for testing data w/ optimal features
        probas_ = classifier.predict_proba(Xts.iloc[:, chromo[-1]])
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
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.2f)' % (mean_auc, std_auc),
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
    plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0))
    plt.show()

    print("Optimal chromosomes for roc curve:", opt_chromos)

# get the cross validated metrics for the proposed model
draw_cv_pr_metrics(clf, skf, X, y, title = 'Proposed Model PR Curve')
draw_cv_roc_curve(clf, skf, X, y, title = 'Proposed Model ROC Curve')
