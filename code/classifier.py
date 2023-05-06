import nltk
import pickle
import random
from scipy.stats import ttest_ind

# Load formal sentences from pickle file
with open('text_data/formal_tokens.pkl', 'rb') as f:
    formal_tokens = pickle.load(f)

# Load informal sentences from pickle file
with open('text_data/informal_tokens.pkl', 'rb') as f:
    informal_tokens = pickle.load(f)

def extract_features(tokens):
    unigrams = tokens
    bigrams = list(nltk.bigrams(tokens))
    pos_tags = [tag for _, tag in nltk.pos_tag(tokens)]

    features = {}
    for word in unigrams:
        features['contains({})'.format(word)] = True
    for bigram in bigrams:
        features['bigram({},{})'.format(bigram[0], bigram[1])] = True
    for pos in pos_tags:
        features['pos({})'.format(pos)] = True
    features['length'] = len(tokens)

    return features

# Filter formal tokens by length
formal_tokens_filtered = [tokens for tokens in formal_tokens if len(tokens) >= 5]

# Get a random sample of formal tokens with the same size as informal tokens
formal_tokens_sample = random.sample(formal_tokens_filtered, len(informal_tokens))

avg_formal_length = sum(len(tokens) for tokens in formal_tokens_sample) / len(formal_tokens_sample)
print("Average length of formal token sequences:", avg_formal_length)

# Calculate average length of informal token sequences
avg_informal_length = sum(len(tokens) for tokens in informal_tokens) / len(informal_tokens)
print("Average length of informal token sequences:", avg_informal_length)

t_stat, p_value = ttest_ind([len(tokens) for tokens in formal_tokens], [len(tokens) for tokens in informal_tokens], equal_var=False)

# Print results of hypothesis test
if p_value < 0.05:
    print("The means are significantly different (p < 0.05)")
    print("p-value: {:.4f}".format(p_value))
else:
    print("The means are not significantly different (p >= 0.05)")
    print("p-value: {:.4f}".format(p_value))

formal_tokens = formal_tokens_sample

formal_feats = [(extract_features(tokens), 'formal') for tokens in formal_tokens]
informal_feats = [(extract_features(tokens), 'informal') for tokens in informal_tokens]


# View the first 3 sequences of tokens for both sets tokenized.
# print(formal_feats[:3])
# print(informal_feats[:3])

# Split the labeled feature sets into training and testing sets (80/20 split)
formal_train_size = int(len(formal_feats) * 0.8)
formal_train_set, formal_test_set = formal_feats[:formal_train_size], formal_feats[formal_train_size:]
informal_train_size = int(len(informal_feats) * 0.8)
informal_train_set, informal_test_set = informal_feats[:informal_train_size], informal_feats[informal_train_size:]

# Concatenate formal and informal training sets
train_set = formal_train_set + informal_train_set
random.shuffle(train_set)

# Split combined training set into training and validation sets
train_size = int(len(train_set) * 0.8)
train_set, val_set = train_set[:train_size], train_set[train_size:]

# Train the Naive Bayes classifier using Laplace smoothing
classifier = nltk.NaiveBayesClassifier.train(train_set, estimator=nltk.LaplaceProbDist)

# Evaluate the accuracy of the classifier on the testing set
accuracy = nltk.classify.accuracy(classifier, val_set) * 100
print("Accuracy: {:.4f}".format(accuracy) + '%')

# Get the top 10 most informative features with likelihood ratios
classifier.show_most_informative_features(10)



