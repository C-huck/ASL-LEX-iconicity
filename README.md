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
- Mean: 0.2589 (p < 0.0001, chance = 1/7; blind baseline = 0.60)
- Std:  0.1241
- Min:  0.0161
- Quart:  [0.1892, 0.2903, 0.3448]
- Max:  0.4194

# Results, lexical features
2-class (Noun vs. Verb) analysis:
- Mean:    0.6003 (p < 0.001, chance = 1/2; p = 1, blind baseline = 0.75)
- Std:     0.1042
- Min:     0.3776
- Quart:   [0.5536, 0.6212, 0.6793]
- Max:     0.7273

- 7-class analysis:
- Mean:    0.2835 (p < 0.001, chance = 1/7; p = 1, blind baseline = 0.60)
- Std:     0.1403
- Min:     0.0328
- Quart:   [0.2093, 0.3185, 0.3815]
- Max:     0.4803

To note, there is a massive class imbalance, with nouns and verbs being ~60% and ~20% of the corpus, respectively, for the 7-class analysis and ~75% and ~25% of the corpus in the 2-class analysis. We address this in three ways:
1. We use a stratified KFold splitter, s.t. the proportion of nouns, verbs[, adjectives, adverbs, numbers, names] are the same across folds
2. We use the ```class_weight``` function of the LogisticRegression classifier that assigns a weight to each class that is inversely proprtional to its frequency. 
3. For  significance testing, we provide both the random baseline (p = 1/number of classes) and the blind baseline (p = frequency of most represented class). For the random baseline, we use a binomial test (```stats.binom_test```) comparing the total number of hits across folds against the total number of items, with a hypothetical random baseline of 1/```clf.classes_```. For the blind baseline, we compute the cummulative mass function of the binomial using p = freq. most represented class (i.e., nouns) as the hypothetical baseline (e.g., the proportion correct answers if the classifier just guessed nouns).

# Interpretation
Unlike results of Monaghan et al. (2007, *Cog. Psycol. 55*), neither visual (iconic) or lexical features of ASL verbs reliably predict category membership. This may be due to the imbalance of lexical classes in the dataset, and/ or the fact that the dataset is only 993 items large (Monaghan et al.'s dataset was multiple thousands times larger). However, both iconic and lexical features *tended* to predict category membership, which suggests better performance as the corpus grows. 
