#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Naive Bayes as described in Speech and Language Processing, 3rd Edition Draft
import math
from typing import List, Dict
from collections import Counter

sample_docs = [
    ['just', 'plain', 'boring'],
    ['entirely', 'predictable', 'and', 'lacks', 'energy'],
    ['no', 'surprises', 'and', 'very', 'few', 'lengths'],
    ['very', 'powerful'],
    ['the', 'most', 'fun', 'film', 'of', 'the', 'summer']]

# negative: 0, positive: 1
sample_labels = [0, 0, 0, 1, 1]
test_doc = [
    ['predictable', 'with', 'no', 'fun'],
    ['powerful', 'messages', 'and', 'exciting', 'film'],
    ['the', 'best', 'movie', 'currently']]


def NaiveBayesTrain(training_docs_list: List[List[str]], 
                    training_labels: List[int]) -> List:
    '''
    Parameters
    ----------
    training_docs_list : List[List[str]]
        a list containing many documents, which are lists of strings
    training_labels : List[int]
        a list of training labels - integers from 0 onwards

    Returns
    -------
    List
        the prior (probability of each class)
        the log-likelihood (probability a word belongs to each class)
        vocabulary (list of unique words across all documents)
        
    Estimates the prior and log-likelihood by simple counting.
    The prior is the ratio of a class to the number of documents.
    The log-likelihood of a word belonging to a class is:
    the count of a word + 1 divided by 
    the count of all words in a class + size of the vocabulary
    '''
    
    # initialization
    n_doc = len(training_docs_list)
    n_class = Counter(training_labels).values()
    p_class = [math.log(c/n_doc) for c in n_class] # prior p
    
    # vocabulary (unique terms)
    vocab = Counter()
    for doc in training_docs_list:
        vocab.update(doc)
    vocabulary = list(vocab)
    V = len(vocabulary)
    
    # group documents according to class
    class_docs = [[]] * len(p_class)
    for i, label in enumerate(training_labels):
        class_docs[label] = class_docs[label] + training_docs_list[i]
    
    # count occurrences of each word in each class
    word_counts = [Counter(d) for d in class_docs]
    
    # count(w', c) + |V|
    denominator = [sum(counter.values()) + V for counter in word_counts]
    
    # add one to ALL counts, even if the word doesn't appear in the class
    likelihood = {}
    for word in vocabulary:
        likelihood[word] = [c[word] + 1 for c in word_counts]
    
    # compute log likelihood
    for k in likelihood.keys():
        for i, d in enumerate(denominator):
            likelihood[k][i] = math.log(likelihood[k][i] / denominator[i])
    return p_class, likelihood, vocabulary
    
def NaiveBayesTest(test_documents: List[List[str]], 
                   prior: List[float], 
                   likelihood: Dict[str, List[float]], 
                   vocabulary: List[str]) -> List[int]:
    '''
    Parameters
    ----------
    test_document : List[List[str]]
        list of tokens for ONE test document
    prior : List[float]
        prior probability of each class
    likelihood : Dict[str, List[float]]
        log-likelihood of each word belonging to each class
    vocabulary : List[str]
        list of unique words across all documents

    Returns
    -------
    List[int]
        the most probable class each test document belongs to
    '''
    predicted_classes = []
    for doc in test_documents:
        sums = {i: prior[i] for i in range(len(prior))}
        for i in range(len(prior)):
            for w in doc:
                if w in vocabulary:
                    sums[i] += likelihood[w][i]
        # argmax that returns class with highest probability
        predicted_classes.append(sorted(sums.items(), 
                                        key = lambda x: x[1], 
                                        reverse = True)[0][0])
    return predicted_classes

def main():
    prior_probabilities, log_likelihood, vocabulary = \
        NaiveBayesTrain(sample_docs, sample_labels)
    predicted_classes = NaiveBayesTest(test_doc, prior_probabilities, 
        log_likelihood, vocabulary)
    print(predicted_classes)
    
if __name__ == '__main__':
    main()
    
