"""
Script loads ASL-LEX data, dummy-codes them, and enters them in a logistic regression model with lexical category as the labels
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from scipy.stats import binom_test

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


fileName = "asl-lex-all-cats.csv"
df = pd.read_csv(fileName)

#Isolate visual features and dummy-code them
visual_cats = ['SignType','MajorLocation','Movement','SelectedFingers','Flexion']
df = pd.get_dummies(df[visual_cats]).join(df['LexicalClass'])

#shuffle data and eliminate features with VIF > 5.0
df_shuf = shuffle(df)
X = calculate_vif_(df_shuf.drop(['LexicalClass'],axis=1))
y = df_shuf['LexicalClass']

#Begin classification
splits = 8
kf = KFold(n_splits=splits,shuffle=True)

predictions = []
ground_truth = []
for train, test in kf.split(X):
  clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',class_weight='balanced').fit(X[train], y[train])
  pred = clf.predict(X[test])
  predictions.append(pred)
  ground_truth.append(y[test])

#compute mean accuracy across all folds
accu = []
accu_raw = []
for i in range(len(predictions)):  
    accu.append(accuracy_score(predictions[i],ground_truth[i]))
    accu_raw.append(accuracy_score(predictions[i],ground_truth[i],normalize=False)) #returns raw hits/ misses, instead of % accurate
print(np.mean(accu))

#calculate per-fold p-value
p =1/float(len(classes))
for i in range(len(accu_raw)):
    print(binom_test(accu_raw[i],len(test),p=p))
