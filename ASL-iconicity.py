"""
Script loads ASL-LEX data, dummy-codes them, and enters them in a logistic regression model with lexical category as the labels
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import binom_test,binom
from sklearn.preprocessing import KBinsDiscretizer

def calculate_vif_(X, thresh=5.0):
    """
    Calculates the variance inflation factor (VIFs) for each predictor in a linear model, 
    and removes features with VIF scores > threshold (default = 5.0). 
    """
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def LogRegCL(X,y):
    splits = 8
    kf = StratifiedKFold(n_splits=splits,shuffle=True)

    #NB: class_weight = 'balanced' accounts for the unequal number of each class
    predictions = []
    ground_truth = []
    for train, test in kf.split(X):
        clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='auto',class_weight='balanced',C=1.0).fit(X[train], y[train])
        pred = clf.predict(X[test])
        predictions.append(pred)
        ground_truth.append(y[test])
        
    #compute mean accuracy across all folds
    accu = []
    accu_raw = []
    for i in range(len(predictions)):  
        accu.append(accuracy_score(predictions[i],ground_truth[i]))
        accu_raw.append(accuracy_score(predictions[i],ground_truth[i],normalize=False)) #returns count hits/ misses for purpose of binomial test
    return accu,accu_raw,clf.classes_

def calculate_p(hits,total,classes):
    p =1.0/classes
    #per-fold p-value
    if type(hits) == list:   
        for i in range(len(hits)):
            print("Fold ",i,": ",round(binom_test(hits[i],round(total/len(hits)),p=p),4))
        #calculate cross-fold p-value
        print("Total: ",round(binom_test(sum(hits),total,p=p),4))
    else:
        print(round(binom_test(hits,total,p=p),4))
        
def binomial_cmf(k, n, p):
    c = 0
    for k1 in range(n+1):
        if k1>=k:
            c += binom.pmf(k1, n, p)
    return c

def summary_stats(accuracy):
    print("Mean:\t",round(np.mean(accuracy),4))
    print("Std:\t",round(np.std(accuracy),4))
    print("Min:\t",round(min(accuracy),4))
    print("Quart:\t",[round(x,4) for x in np.quantile(accuracy,[0.25,0.5,0.75])])
    print("Max:\t",round(max(accuracy),4))

#read in data
fileIn = "SignData.csv"
df = pd.read_csv(fileIn)

#discretize sign length
n_bins = 3 #number of bins to sort sign length into. 3 might mean "high", "medium", and "low" length
sl_to_bin = np.array(df['SignLength(ms)']).reshape(-1,1)
sl_bin = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile').fit_transform(sl_to_bin)
df_SL = pd.DataFrame(data=sl_bin,columns=['SL_long','SL_med','SL_short'])
df_SL = df_SL.drop(['SL_long'],axis=1) #drop one category to avoid singular matrix; SL_long is the reference category

#Isolate visual features and dummy-code them; drop first category set to True to avoid singular matrix
#join them in one dataframe with class labels
visual_cats = ['SignType','MajorLocation','Movement','SelectedFingers','Flexion']
df = pd.get_dummies(df[visual_cats],drop_first=True).join(df['LexicalClass'])

#merge categorical and discretized dataframes
df = pd.concat([df,df_SL],axis=1)

#7-category or noun-vs.-verb analysis
#df = df[df['LexicalClass'].isin(['Noun','Verb'])] #uncomment to run noun-vs.-verb analysis

#As a backup, eliminate features with VIF > 5.0
X = calculate_vif_(df.drop(['LexicalClass'],axis=1)) #comment to not calculate vif
#X = df.drop(['LexicalClass'],axis=1) #uncomment to calculate vif
y = df['LexicalClass']
X,y = np.array(X),np.array(y)

#Begin classification
accu,accu_raw,classes= LogRegCL(X,y)
calculate_p(accu_raw,len(X),len(classes))
summary_stats(accu)

most_freq_class = df['LexicalClass'].value_counts().max()
p = most_freq_class/len(df)
binomial_cmf(sum(accu_raw),len(df),p=p)
