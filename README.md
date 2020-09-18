# NLP_Moderator
Using Natural Language Processing Machine Learning Algorithms for Toxicity / Targeted Insult Detection

## NLP Algorithms

Multiple algorithms will be tested. Only the Naive Bayes Classifier has been used as of now.

## Data Set

The data set is from Google Jigsaw's Kaggle challenge for content moderation (I am not participating in the competition, just using the available data). The data is stored in a .csv file with values assigned as follows: [toxic or non-toxic], [date], [message]. The sample size is 3947 messages and corresponding classifications.

## Naive Bayes Classifier

The Naive Bayes Classifier conducts frequency analysis of word occurances, then compares the data to the classification (insult or not_insult) for training. To test, the classifier once again conducts a word-occurance frequency analysis, and predicts the classification based on which word frequencies are commonly associated with which classifications. For example, profanities and slurs tend to occur much more frequently in toxic messages than in non-toxic ones.

The current success rate with the training size 3000, and test size 947 is approximately 70% (+/- 3%). Adjustments will be made in an attempt to more accurately classify messsages.
