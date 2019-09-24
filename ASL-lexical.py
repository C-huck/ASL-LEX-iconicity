"""
Same as ASL-iconicity.py, except that only lexical features (not visual) are included in the model. 
Model predictors in this case are continuous, not categorical, and require standardization.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import binom_test,binom
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

def calculate_vif_(X, thresh=5.0):
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
    kf = StratifiedKFold(n_splits=splits,shuffle=False)
    #kf = KFold(n_splits=splits,shuffle=False)

    #NB: class_weight = 'balanced' accounts for the unequal number of each class
    #multi_class = 'auto'; uses 'multinmial' if n_classes > 2, else 'ovr' 
    predictions = []
    ground_truth = []
    for train, test in kf.split(X,y):
        clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='auto',class_weight='balanced').fit(X[train], y[train])
        pred = clf.predict(X[test])
        predictions.append(pred)
        ground_truth.append(y[test])
        print(Counter(y[train]),Counter(y[test]))
        print(Counter(pred))
        
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
fileIn = "asl-lex-all-cats.csv"
df = pd.read_csv(fileIn)

#Isolate lexical features and dummy-code them
lexical_cats = ['SignFrequency(M)','Iconicity(M)','MinimalNeighborhoodDensity', 'MaximalNeighborhoodDensity',
       'Parameter-BasedNeighborhoodDensity', 'SignTypeFrequency',
       'MajorLocationFrequency', 'MinorLocationFrequency',
       'SelectedFingersFrequency', 'FlexionFrequency', 'MovementFrequency',
       'HandshapeFrequency','LexicalClass']
df = df[lexical_cats]

#Uncomment for noun-vs.-verb analysis
#df = df[df['LexicalClass'].isin(['Noun','Verb'])]

#Prepare data vectors for classification
X = df.drop(['LexicalClass'],axis=1)
y = df['LexicalClass']
X,y = np.array(X),np.array(y)

#Fill in missing/ NaN values with column mean
#Then standardize by removing the mean and scaling to unit variance
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)
X = StandardScaler().fit_transform(X)

#Begin classification
accu,accu_raw,classes= LogRegCL(X,y)

#Calculate significance using random baseline
calculate_p(accu_raw,len(X),len(classes))

#Summary of classifier performance
summary_stats(accu)

#Calculate significance using blind baseline
most_freq_class = df['LexicalClass'].value_counts().max()
p = most_freq_class/len(df)
binomial_cmf(sum(accu_raw),len(df),p=p)
