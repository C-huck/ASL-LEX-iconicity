# ASL-LEX-iconicity
Is lexical category information apparent in the form of ASL signs?

Script isolates visual features from the ASL-LEX corpus and correlates them with lexical class labels using an 8-fold leave-one-out Logistic Regression paradigm. 

# Results, visual features
2-class (Noun vs. Verb) analysis:
- Mean:    0.6602 (p <0.001, chance = 1/2; p = 1, blind baseline = 0.75)
- Std:     0.0424
- Min:     0.6122
- Quart:   [0.6224, 0.6515, 0.6869]
- Max:     0.7475

7-class analysis:
- Mean: 0.2589 (p < 0.0001, chance = 1/7; )
- Std:  0.1241
- Min:  0.0161
- Quart:  [0.1892, 0.2903, 0.3448]
- Max:  0.4194

# Results, lexical features

To note, there is a massive class imbalance, with nouns and verbs being ~60% and ~20% of the corpus, respectively, for the 7-class analysis and ~75% and ~25% of the corpus in the 2-class analysis. We address this in three ways:
1. We use a stratified KFold splitter, s.t. the proportion of nouns, verbs[, adjectives, adverbs, numbers, names] are the same across folds
2. We use the ```class_weight``` function of the LogisticRegression classifier that assigns a weight to each class that is inversely proprtional to its frequency. 
3. For  significance testing, we provide both the random baseline (p = 1/number of classes) and the blind baseline (p = frequency of most represented class). 
