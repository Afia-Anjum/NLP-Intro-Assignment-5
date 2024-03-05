'''
https://pythonmachinelearning.pro/text-classification-tutorial-with-naive-bayes/
https://medium.com/@makcedward/nlp-pipeline-stop-words-part-5-d6770df8a936 --> stop word removal guide
https://www.programiz.com/python-programming/writing-csv-files --> writing CSV output file guide
'''
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
import re
import string
import math
import operator
import nltk
import csv

def clean(s):
    translator = str.maketrans("", "", string.punctuation)
    return s.translate(translator)

def tokenize(text, remove_stop_words=False):
    text = clean(text).lower()
    if remove_stop_words:
        nltk_stopwords = nltk.corpus.stopwords.words('english')
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [token for token in tokens if not token in nltk_stopwords]
        return tokens
    return re.split("\W+", text)

def get_word_counts(words):
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0.0) + 1.0
    return word_counts

## Takes in a filename and returns a dictionary of two items: a list of all categories and a list of the correspinding texts ##
def read_csv_file(filename):
    csv_dict = defaultdict(list)
    with open(filename, encoding='utf-8') as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                csv_dict[k].append(v) # append the value into the appropriate list

    return csv_dict

## Takes a list of all categories and all the corresponding texts, and returns five separate lists of texts for each of the five categories ##
def split_category_texts(categories, texts):
    tech, business, entertainment, politics, sports = ([], [], [], [], [])
    for i in range(len(categories)):
        category = categories[i]
        text = texts[i]
        if category == 'tech':
            tech.append(text)
        elif category == 'business':
            business.append(text)
        elif category == 'entertainment':
            entertainment.append(text)
        elif category == 'politics':
            politics.append(text)
        elif category == 'sport':
            sports.append(text)

    return tech, business, entertainment, politics, sports

## Takes in a confusion matrix and computes and returns the precision and recall for that category if cat_idx is specified; otherwise it computes these for all categories ## 
def compute_precision_recall(matrix, cat_idx=-1):
    if cat_idx != -1:
        tp = matrix[cat_idx, cat_idx]
        fp = np.sum(matrix[:,cat_idx]) - tp
        fn = np.sum(matrix[cat_idx,:]) - tp
    else:
        # Loop over the entire matrix for microaverage computations
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(matrix)):
            tp += matrix[i,i]
            fp += np.sum(matrix[:,i]) - matrix[i,i]
            fn += np.sum(matrix[i,:]) - matrix[i,i]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall

## Training algorithm implemented as described in Fig. 4.2 of the textbook ##
## Params is a dict different NB model techniques to test ##
def trainNB(categories, texts, category_labels, params):
    N = len(texts) # No. of training documents

    # Separate lists of texts for each category
    tech, business, entertainment, politics, sport = split_category_texts(categories, texts)

    category_texts = [tech, business, entertainment, politics, sport]

    log_class_priors = {}
    word_counts = {}    # Holds dictionary of words and their counts across all texts for each category
    vocab = set()       # Holds the vocabulary of words found in all training texts
    for i in range(len(category_texts)):
        category = category_labels[i]
        texts = category_texts[i]

        # Calculating (log) prior probabilities
        log_class_priors[category] = math.log(len(texts) / N)
        
        # Calculating word counts for the given class and building training vocab
        word_counts[category] = {}
        for j in range(len(texts)):
            text = texts[j]
            word_tokens = tokenize(text, params['remove_stop_words'])
            w_counts = get_word_counts(word_tokens)

            for word, count in w_counts.items():
                if word not in vocab:
                    vocab.add(word)
                word_counts[category][word] = word_counts[category].get(word, 0.0) + count

    # Calculating loglikelihood for each word in the vocubulary
    log_likelihoods = {}
    for category in category_labels:
        log_likelihoods[category] = {}
        for word in vocab:
            log_likelihoods[category][word] = math.log((word_counts[category].get(word, 0) + 1) / (sum(word_counts[category].values()) + len(vocab)))

    return log_class_priors, log_likelihoods, vocab, word_counts

## Testing algorithm implemented as described in Fig. 4.2 of the textbook ##
## Params is a dict different NB model techniques to test ##
def testNB(texts, logpriors, loglikelihoods, vocab, category_labels, word_counts, params):
    results = []
    for i in range(len(texts)):
        logprob_scores = {}
        for category in category_labels:
            logprob_scores[category] = logpriors[category]

            words = tokenize(texts[i], params['remove_stop_words'])
            for word in words:
                if word not in vocab:
                    if params['remove_unknown_words']:
                        logprob_scores[category] += math.log(1 / (sum(word_counts[category].values()) + len(vocab)))
                    else: continue
                else:
                    logprob_scores[category] += loglikelihoods[category][word]

        chosen_category = max(logprob_scores.items(), key=operator.itemgetter(1))[0]
        results.append(chosen_category)

    return results

## k-fold cross-validation ##
# Xtrain - texts data to partition
# Ytrain - category labels to partition
# K - number of folds
# parameters - a list of parameter dictionaries to test
def cross_validate(Xtrain, Ytrain, K, parameters, category_labels):
    all_accs = np.zeros((len(parameters), K))

    subset_size = int(len(Xtrain) / K)
    for k in range(K):
        # Compute slices for validation and train sets
        val_indices = slice(k*subset_size, k*subset_size + subset_size)
        train_indices1 = slice(k*subset_size)
        train_indices2 = slice((k+1)*subset_size, None)

        # Printing data slices
        print("Fold " + str(k) + ": ")
        print("Data slices in the form: slice(start_index, stop_index, step_size)")
        print("Validation set: " + str(val_indices))
        print("Training set: " + str(train_indices1) + " & " + str(train_indices2))
        print()

        X_validation_set = Xtrain[val_indices]
        Y_validation_set = Ytrain[val_indices]
        X_training_set = Xtrain[train_indices1] + Xtrain[train_indices2]
        Y_training_set = Ytrain[train_indices1] + Ytrain[train_indices2]

        # Running the model for each set of params and saving the accuracy scores
        for i, params in enumerate(parameters):
            l_priors, l_likelihoods, vocab, word_counts = trainNB(Y_training_set, X_training_set, category_labels, params)
            results = testNB(X_validation_set, l_priors, l_likelihoods, vocab, category_labels, word_counts, params)

            accuracy = accuracy_score(Y_validation_set, results)
            all_accs[i,k] = accuracy

    # Go through all accuracies and pick the best one
    avg_accs = np.mean(all_accs, axis=1)
    best_params = parameters[0]
    best_acc = 0
    for i, params in enumerate(parameters):
        avg_acc = avg_accs[i]
        print('Cross validate parameters:', params)
        print('average accuracy:', avg_acc)

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_params = params

    return best_params

## Helper function to see the inaccurately classfied texts ##
def analyze_results(actual, predicted, texts):
    for i in range(len(texts)):
        if actual[i] != predicted[i]:
            print("actual:", actual[i])
            print("predicted:", predicted[i])
            print(texts[i])

## Takes in a filename, list of actual labels, list of predicted labels, lists of texts, and writes a csv file in the same folder ##
def create_output_file(filename, actual, predicted, texts):
    with open('output_' + filename + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["original_label", "assigned_label", "text"])

        for i in range(len(texts)):
            row = [actual[i], predicted[i], texts[i]]
            writer.writerow(row)  

def main():
    category_labels = ('tech', 'business', 'entertainment', 'politics', 'sport')

    # TRAIN
    train = read_csv_file('a5_data_students/trainBBC.csv')
    train_categories = train['\ufeffcategory']
    train_texts = train['text']
    
    # Set of training techniques to try with NB
    params = [
        { 'remove_stop_words': False, 'remove_unknown_words': False },
        { 'remove_stop_words': True, 'remove_unknown_words': False },
        { 'remove_stop_words': False, 'remove_unknown_words': True },
        { 'remove_stop_words': True, 'remove_unknown_words': True }
    ]

    # Best params as a result of cross-validation
    best_params = cross_validate(train_texts, train_categories, 3, params, category_labels)
    print("\nThe best parameters as a result of cross-validation are:", best_params)

    l_priors, l_likelihoods, vocab, word_counts = trainNB(train_categories, train_texts, category_labels, best_params)

    # TEST
    test = read_csv_file('a5_data_students/testBBC.csv')
    test_categories = test['\ufeffcategory']
    test_texts = test['text']

    results = testNB(test_texts, l_priors, l_likelihoods, vocab, category_labels, word_counts, best_params)
    conf_matrix = confusion_matrix(test_categories, results, labels=category_labels)

    # Evaluation metrics
    print("\nAccuracy:", accuracy_score(test_categories, results))
    print("Confusion matrix (predicted labels on horizontal axis; actual labels on vertical; labels from left to right (or top to bottom) follow: tech, business, entertainment, politics, sport)")
    print(conf_matrix)

    # Computing micro and macro precision scores
    micro_precision = compute_precision_recall(conf_matrix)[0]

    prec_sum = 0
    for i in range(len(conf_matrix)):
        scores = compute_precision_recall(conf_matrix, i)
        precision = scores[0]
        recall = scores[1]
        print(category_labels[i] + ":", "precision", precision, ", recall", recall)
        prec_sum += precision
    macro_precision = prec_sum / len(conf_matrix)

    print("\nmicro_precision:", micro_precision)
    print("macro_precision", macro_precision)

    # Analyze misclassifications (uncomment below to analyze misclassifications)
    # analyze_results(test_categories, results, test_texts)

    # EVALUATION
    eval_set = read_csv_file('a5_data_students/evalBBC.csv')
    eval_texts = eval_set['text']
    eval_results = testNB(eval_texts, l_priors, l_likelihoods, vocab, category_labels, word_counts, best_params)

    # Generate output files
    create_output_file('testBBC', test_categories, results, test_texts)
    create_output_file('evalBBC', ['' for i in range(len(eval_results))], eval_results, eval_texts)
    
main()