"""
Same as ASL-iconicity.py, except that only lexical features (not strictly visual) are included in the model. 
Model predictors in this case are continuous, not categorical, and require standardization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from scipy.stats import binom_test
from sklearn.preprocessing import StandardScaler, QuantileTransformer

#read in data
fileName = "asl-lex-all-cats.csv"
df = pd.read_csv(fileName)

#Isolate lexical features and dummy-code them
lexical_cats = ['SignFrequency(M)','Iconicity(M)','MinimalNeighborhoodDensity', 'MaximalNeighborhoodDensity',
       'Parameter-BasedNeighborhoodDensity', 'SignTypeFrequency',
       'MajorLocationFrequency', 'MinorLocationFrequency',
       'SelectedFingersFrequency', 'FlexionFrequency', 'MovementFrequency',
       'HandshapeFrequency','LexicalClass']
df = df[lexical_cats] #includes all lexical categories: noun, verb, adjective, adverb, number, minor, & name
#df = df[df['LexicalClass'].isin(['Verb','Noun'])] #uncomment to select only nouns & verbs
classes = set(df['LexicalClass'])

#fill and NaN cells
#two items do not have iconicity scores, so we fill them with the average iconicity score
#otherwise NaN is filled with 0
df['Iconicity(M)'].fillna(np.mean(df['Iconicity(M)'))
df.fillna(0)

#shuffle data and convert to arrays
df_shuf = shuffle(df)
X = np.array(df_shuf.drop(['LexicalClass'],axis=1))
y = np.array(df_shuf['LexicalClass'])

#Choose how to preprocess data
#X = StandardScaler().fit_transform(X)
X = QuantileTransformer().fit_transform(X)

#Begin classification
splits = 8
kf = KFold(n_splits=splits,shuffle=True)

predictions = []
ground_truth = []
for train, test in kf.split(X):
  """
  Choose classifier. 
  For multiclass option, choose lbfgs solver and multi_class='multinomial'
  For binary options (e.g., noun vs. verb), choose liblinear solver and multi_class='ovr' (default)
  To-do: implement one-vs.-rest strategy
  """
  #clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',class_weight='balanced').fit(X[train], y[train])
  clf = LogisticRegression(random_state=0, solver='liblinear',class_weight='balanced').fit(X[train], y[train])
  pred = clf.predict(X[test])
  predictions.append(pred)
  ground_truth.append(y[test])

accu = []
accu_raw = []
for i in range(len(predictions)):  
    accu.append(accuracy_score(predictions[i],ground_truth[i]))
    accu_raw.append(accuracy_score(predictions[i],ground_truth[i],normalize=False))
print("Mean accuracy: ",np.mean(accu))

p =1/float(len(classes))
print("Chance accuracy: ",p)
raw_accu = []
for i in range(len(accu_raw)):
    print("Fold "+str(i)+" significance: ",binom_test(accu_raw[i],len(test),p=p))
    raw_accu.append(accu_raw[i])
print("Total significance: ",binom_test(sum(raw_accu),len(df),p=p))
