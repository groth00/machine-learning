#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.clustering import LDA

import sparknlp
from sparknlp import DocumentAssembler, Finisher
from sparknlp.annotator import SentenceDetector, Tokenizer,\
    LemmatizerModel, Normalizer

# https://www.kaggle.com/rowhitswami/nips-papers-1987-2019-updated
papers = pd.read_csv('nips/papers.csv')\
    .dropna(subset = ['full_text'], axis = 0)
papers = papers.fillna(value = {'abstract': '-'})

spark = sparknlp.start()

schema = StructType([
    StructField('source_id', IntegerType()),
    StructField('year', IntegerType()),
    StructField('title', StringType()),
    StructField('abstract', StringType()),
    StructField('texts', StringType())
    ])

df = spark.read.option("quote", "\"")\
    .option("escape", "\"")\
    .option("header", True)\
    .csv(path = 'nips/papers.csv', schema = schema,
         ignoreTrailingWhiteSpace = True, multiLine = True,
         unescapedQuoteHandling = 'RAISE_ERROR')

# create sparkdf to verify output of the tokenization, pipeline, etc.
def create_example(pandas_df, schema, n):
    it = pandas_df.itertuples(index = False, name = None)
    example = spark.createDataFrame(
        [next(it) for i in range(n)], schema = schema)
    return example

assembler = DocumentAssembler()\
    .setInputCol('texts').setOutputCol('document')
    
tokenizer = Tokenizer()\
    .setInputCols(['document']).setOutputCol('tokens')
    
lemmatizer = LemmatizerModel.pretrained()\
    .setInputCols(['tokens']).setOutputCol('lemmatized')
    
normalizer = Normalizer()\
    .setInputCols(['lemmatized']).setOutputCol('normalized')
    
finisher = Finisher()\
    .setInputCols(['normalized']).setOutputCols(['normalized_tokens'])
    
stopwords = StopWordsRemover().loadDefaultStopWords('english')
NEW_STOPWORDS = ['et', 'al', 'cid', 'arXiv']
stopwords.extend(NEW_STOPWORDS)

stop = StopWordsRemover()\
    .setInputCol('normalized_tokens').setOutputCol('stopwords')\
    .setStopWords(stopwords)

pipeline = Pipeline().setStages(
    [assembler, tokenizer, lemmatizer, normalizer, finisher, stop])

# may want to shorten the vocabulary (currently is ~260,000)
tf = CountVectorizer().setInputCol('stopwords').setOutputCol('tf')
idf = IDF().setInputCol('tf').setOutputCol('tfidf')
lda = LDA().setFeaturesCol('tfidf').setK(10).setSeed(123)

final_pipeline = Pipeline().setStages([
    pipeline,
    tf,
    idf,
    lda])

# forgot to time the entire pipeline, took ~30+ minutes or so?
# look into deploying this script elsewhere to quickly test other params
model = final_pipeline.fit(df)
tf_model = model.stages[-3]
lda_model = model.stages[-1]
vocabulary = tf_model.vocabulary 

# describeTopics() returns..
# DataFrame[topic: int, termIndices: array<int>, termWeights: array<double>]
# use termIndices and tf_model.vocabulary to look up words
# also need to figure out how to assign each doc to a topic
pd_50terms = lda_model.describeTopics(50).toPandas()

# after training; extra words to remove 
additional_stopwords = ['et', 'al', 'cid', 'arXiv']

# temporary solution to filter additional vocabulary after training
def show_topics_filter_words(pandas_df, vocabulary, stopwords, n_words):
    for row in pandas_df.itertuples(name = None, index = False):
        print(f'Topic {row[0] + 1}')
        unfiltered_words = 0
        for index, weight in zip(row[1], row[2]):
            if unfiltered_words > n_words:
                break
            word = vocabulary[index]
            if len(word) != 1 and word not in stopwords:
                unfiltered_words += 1
                print(f'{vocabulary[index]:>25}, {weight:.5f}')
        print ('-' * 50)

show_topics_filter_words(pd_50terms, vocabulary, additional_stopwords, 20)

# -------------------------------- Other Notes
# need to remove symbols (single-length tokens) using regex

# possible words to stem: privacy/private, differentially/differential, 
# submodular/submodularity

# i noticed a duplicate entry due to capitalization, but
# converting all to lowercase will ruin acronyms and initialisms..

# consider looking at a bigram/trigram model (in spark/gensim)
# --------------------------------
'''
Topic 1
                   causal, 0.00175
              variational, 0.00154
                   latent, 0.00144
                    label, 0.00134
                 fairness, 0.00126
                  network, 0.00114
            classiﬁcation, 0.00109
                 gradient, 0.00108
                inference, 0.00107
                classiﬁer, 0.00103
                     loss, 0.00101
                    graph, 0.00100
                     deep, 0.00099
                  dataset, 0.00096
                 training, 0.00091
               generative, 0.00089
                    score, 0.00088
                    image, 0.00086
                     risk, 0.00084
                     task, 0.00084
                    MNIST, 0.00083
--------------------------------------------------
Topic 2
                   kernel, 0.00198
                   convex, 0.00161
                   matrix, 0.00161
                  Theorem, 0.00132
                     norm, 0.00119
                   tensor, 0.00109
                estimator, 0.00109
                    Lemma, 0.00103
                 gradient, 0.00100
                    graph, 0.00099
                   sparse, 0.00098
                     bind, 0.00098
                   bounds, 0.00089
               regression, 0.00089
              convergence, 0.00089
                 recovery, 0.00087
                       Rd, 0.00086
             optimization, 0.00084
                     rank, 0.00084
                      log, 0.00084
                Algorithm, 0.00081
--------------------------------------------------
Topic 3
                    graph, 0.00264
                     hash, 0.00181
                    query, 0.00180
                     node, 0.00169
                   kernel, 0.00164
                    image, 0.00156
                     word, 0.00154
                  cluster, 0.00147
                     tree, 0.00130
                    label, 0.00127
                    embed, 0.00125
                      cue, 0.00116
                      tag, 0.00116
                     edge, 0.00116
                   vertex, 0.00111
                 saliency, 0.00107
                   object, 0.00104
                 neighbor, 0.00099
                      DBM, 0.00099
                    parse, 0.00093
                 distance, 0.00092
--------------------------------------------------
Topic 4
                     game, 0.00321
                   regret, 0.00306
                   player, 0.00263
                   policy, 0.00233
                     node, 0.00197
                    round, 0.00165
                   reward, 0.00153
                   action, 0.00151
                    agent, 0.00149
                     user, 0.00136
                    buyer, 0.00131
                    price, 0.00126
                  learner, 0.00125
                   expert, 0.00120
                 strategy, 0.00116
                     tree, 0.00113
                     item, 0.00108
                  auction, 0.00103
                  revenue, 0.00098
                     Nash, 0.00095
                    graph, 0.00094
--------------------------------------------------
Topic 5
                    image, 0.00459
                    layer, 0.00296
                   object, 0.00290
              adversarial, 0.00250
                  network, 0.00198
                     deep, 0.00195
            convolutional, 0.00189
            discriminator, 0.00161
                    CIFAR, 0.00152
                    video, 0.00151
             architecture, 0.00151
                   attack, 0.00150
              convolution, 0.00149
                 training, 0.00142
             segmentation, 0.00141
                  dataset, 0.00140
                   ResNet, 0.00135
                     CVPR, 0.00133
                      CNN, 0.00131
                    train, 0.00128
                 sentence, 0.00127
--------------------------------------------------
Topic 6
                  cluster, 0.00212
                  network, 0.00139
               classifier, 0.00131
                     node, 0.00122
                     unit, 0.00122
                     hide, 0.00112
                  mixture, 0.00104
                posterior, 0.00099
           classification, 0.00097
                    field, 0.00097
                inference, 0.00097
                    first, 0.00097
                   define, 0.00097
                     tree, 0.00095
                   filter, 0.00095
                       EM, 0.00093
                  feature, 0.00092
                    layer, 0.00091
                 Bayesian, 0.00090
                  message, 0.00085
                      HMM, 0.00085
--------------------------------------------------
Topic 7
                   policy, 0.00579
                   neuron, 0.00456
                    spike, 0.00451
                   reward, 0.00424
                      arm, 0.00383
                   action, 0.00318
                    agent, 0.00313
                   bandit, 0.00244
                 stimulus, 0.00213
                     cell, 0.00200
                   regret, 0.00194
                 synaptic, 0.00184
            reinforcement, 0.00168
                    state, 0.00166
                  synapse, 0.00162
                       RL, 0.00149
              environment, 0.00141
                 activity, 0.00140
                     fire, 0.00136
              exploration, 0.00130
                 dynamics, 0.00124
--------------------------------------------------
Topic 8
                  privacy, 0.00592
                  private, 0.00414
           differentially, 0.00166
              PACBayesian, 0.00135
                     loss, 0.00128
               hypothesis, 0.00125
             differential, 0.00117
                 PACBayes, 0.00115
                     AROW, 0.00110
                     cidi, 0.00109
                     risk, 0.00109
                   margin, 0.00107
                    party, 0.00098
                 protocol, 0.00096
                  Privacy, 0.00092
                    query, 0.00092
                       DP, 0.00088
                    label, 0.00088
                mechanism, 0.00087
                       yi, 0.00083
                      MHE, 0.00082
--------------------------------------------------
Topic 9
               submodular, 0.00586
                    topic, 0.00491
                 document, 0.00328
                      BCI, 0.00206
                     item, 0.00198
                   market, 0.00177
                      LDA, 0.00168
                      EEG, 0.00160
                     word, 0.00147
                   greedy, 0.00129
                   anchor, 0.00125
                      CSP, 0.00122
                 PageRank, 0.00103
                   trader, 0.00103
                  cluster, 0.00099
               Submodular, 0.00096
                   vertex, 0.00093
                    movie, 0.00081
            submodularity, 0.00080
                  Nystrom, 0.00074
                     PLSA, 0.00074
--------------------------------------------------
Topic 10
                       GP, 0.00180
                     ADMM, 0.00128
                  circuit, 0.00119
                  ZOAdaMM, 0.00095
                     GNNs, 0.00093
                   attack, 0.00093
                      GNN, 0.00090
              equivariant, 0.00084
                 proximal, 0.00082
                      PCR, 0.00079
                       VB, 0.00079
                    LISTA, 0.00076
                  cidRpug, 0.00075
                    lasso, 0.00072
                  residue, 0.00069
                posterior, 0.00069
                     GOSS, 0.00066
                     GBDT, 0.00066
                       ZO, 0.00063
                 backdoor, 0.00061
                     PLDS, 0.00059
--------------------------------------------------
'''