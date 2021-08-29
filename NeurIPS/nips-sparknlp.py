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
         unescapedQuoteHandling = 'RAISE_ERROR').persist()

# create sparkdf to verify output of the tokenization, pipeline, etc.
def create_example(pandas_df, schema, n):
    it = pandas_df.itertuples(index = False, name = None)
    example = spark.createDataFrame(
        [next(it) for i in range(n)], schema = schema)
    return example

assembler = DocumentAssembler()\
    .setInputCol('texts').setOutputCol('document')
    
tokenizer = Tokenizer()\
    .setInputCols(['document']).setOutputCol('tokens')\
    .setMinLength(2) # remove symbols (single length tokens)
    
lemmatizer = LemmatizerModel.pretrained()\
    .setInputCols(['tokens']).setOutputCol('lemmatized')
    
normalizer = Normalizer()\
    .setInputCols(['lemmatized']).setOutputCol('normalized')
    
finisher = Finisher()\
    .setInputCols(['normalized']).setOutputCols(['normalized_tokens'])
    
stopwords = StopWordsRemover().loadDefaultStopWords('english')
# stopwords I found after training
NEW_STOPWORDS = ['et', 'al', 'cid', 'arXiv', 'yi']
stopwords.extend(NEW_STOPWORDS)

stop = StopWordsRemover()\
    .setInputCol('normalized_tokens').setOutputCol('stopwords')\
    .setStopWords(stopwords)

pipeline = Pipeline().setStages(
    [assembler, tokenizer, lemmatizer, normalizer, finisher, stop])

# shorten the vocabulary using VocabSize (default is more than 200,000)
tf = CountVectorizer()\
    .setInputCol('stopwords')\
    .setOutputCol('tf')\
    .setVocabSize(50_000)
idf = IDF().setInputCol('tf').setOutputCol('tfidf')

# may want to experiment with different K (topics)
lda = LDA().setFeaturesCol('tfidf').setK(10).setSeed(123)\
    .setTopicDistributionCol('topicDistributionCol')

final_pipeline = Pipeline().setStages([
    pipeline,
    tf,
    idf,
    lda])

# look into deploying this script elsewhere to quickly test other params
# for me, fitting takes 35 minutes
from time import time

t0 = time()
model = final_pipeline.fit(df)
print(time() - t0)

tf_model = model.stages[-3]
lda_model = model.stages[-1]
vocabulary = tf_model.vocabulary 

# this section assigns documents to one of the K topics
transformed = model.transform(df).persist()
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import expr
# need to convert DenseVector to array
t = transformed.withColumn(
    'topicCol', 
    vector_to_array('topicDistributionCol'))
t2 = t.withColumn(
    colName = 'rowmax', 
    col = expr('array_position(topicCol, array_max(topicCol))'))
# t2.select(f.approx_count_distinct('rowmax')).show()
final = t2.select(['year', 'title', 'rowmax'])
# saving took ~10 minutes
final.write.format('csv').save('lda_output/fixed/') 

# describeTopics() returns..
# DataFrame[topic: int, termIndices: array<int>, termWeights: array<double>]
# use termIndices and tf_model.vocabulary to look up words
pd_50terms = lda_model.describeTopics(50).toPandas()
pd_50terms.to_csv('lda_fitted.csv')

# after training; extra words to remove 
additional_stopwords = ['et', 'al', 'cid', 'arXiv']

# temporary solution to filter additional vocabulary after training
def show_topics_filter_words(pandas_df, vocabulary, stopwords, n_words):
    for row in pandas_df.itertuples(name = None, index = False):
        print ('-' * 50)
        print(f'Topic {row[0] + 1}')
        unfiltered_words = 0
        for index, weight in zip(row[1], row[2]):
            if unfiltered_words > n_words:
                break
            word = vocabulary[index]
            if len(word) != 1 and word not in stopwords:
                unfiltered_words += 1
                print(f'{vocabulary[index]:>25}, {weight:.5f}')

show_topics_filter_words(pd_50terms, vocabulary, stopwords = [], n_words = 20)

# -------------------------------- Other Notes
# stem these?: privacy/private, differentially/differential, 
# submodular/submodularity

# different amount of topics

# consider looking at a bigram model (in spark/gensim)
# or a different model?: non-negative matrix factorization

# visualizations: wordcloud, topics assigned per year, 
# word importance per topic
# --------------------------------

# with vocab = 50,000, length of tokens at least 2
'''
--------------------------------------------------
Topic 1
                    graph, 0.00315
                   tensor, 0.00295
                   vertex, 0.00249
                       GP, 0.00207
                  cluster, 0.00204
                     node, 0.00167
                   matrix, 0.00167
                   kernel, 0.00160
                     edge, 0.00157
               submodular, 0.00154
                 recovery, 0.00139
                diffusion, 0.00126
                  coreset, 0.00122
            approximation, 0.00110
                  Theorem, 0.00110
                   sparse, 0.00104
               covariance, 0.00104
                 gradient, 0.00103
                   copula, 0.00102
                     rank, 0.00101
                Algorithm, 0.00100
--------------------------------------------------
Topic 2
                 fairness, 0.00334
                    graph, 0.00314
                     node, 0.00281
                   kernel, 0.00255
                  speaker, 0.00228
                    audio, 0.00152
                     atom, 0.00151
                     risk, 0.00150
                estimator, 0.00150
                attribute, 0.00146
               embeddings, 0.00131
                   domain, 0.00123
                 molecule, 0.00118
                     fair, 0.00114
                    embed, 0.00107
            classiﬁcation, 0.00107
                    voice, 0.00106
                     tree, 0.00104
                      MLN, 0.00099
                  dataset, 0.00097
                   entity, 0.00097
--------------------------------------------------
Topic 3
                     tree, 0.00353
                    topic, 0.00286
                    label, 0.00271
              variational, 0.00269
                inference, 0.00194
                     node, 0.00174
                posterior, 0.00167
                    Gibbs, 0.00161
                 document, 0.00161
                Dirichlet, 0.00154
                  latexit, 0.00154
                   latent, 0.00149
                  mixture, 0.00144
                unlabeled, 0.00143
                  sampler, 0.00143
                multitask, 0.00140
                     word, 0.00133
                      LDA, 0.00127
            classiﬁcation, 0.00122
                     leaf, 0.00108
                     SWAP, 0.00107
--------------------------------------------------
Topic 4
                   object, 0.00426
                    image, 0.00359
                   causal, 0.00327
                    graph, 0.00288
                     node, 0.00242
                     edge, 0.00238
             segmentation, 0.00233
                      DAG, 0.00212
                      VQA, 0.00194
                    parse, 0.00175
                   parent, 0.00168
                  contour, 0.00165
                  message, 0.00160
                   motion, 0.00158
                  segment, 0.00149
                   vertex, 0.00146
                detection, 0.00144
                     tree, 0.00142
                    scene, 0.00135
                    human, 0.00135
                    pixel, 0.00132
--------------------------------------------------
Topic 5
                    image, 0.00300
                   kernel, 0.00268
                  cluster, 0.00250
                   matrix, 0.00202
                   object, 0.00192
                    label, 0.00178
               classifier, 0.00162
                     rank, 0.00155
                  feature, 0.00150
                 manifold, 0.00135
                      SVM, 0.00130
                   latent, 0.00127
                       EM, 0.00121
                    query, 0.00102
                    shape, 0.00100
              recognition, 0.00100
                  dataset, 0.00099
                      PCA, 0.00097
                 training, 0.00096
                     face, 0.00095
                     word, 0.00092
--------------------------------------------------
Topic 6
                   neuron, 0.00627
                    spike, 0.00621
                 stimulus, 0.00365
                     cell, 0.00348
                 synaptic, 0.00226
                 activity, 0.00222
                 response, 0.00219
                  synapse, 0.00217
                  circuit, 0.00209
                  privacy, 0.00198
                     fire, 0.00179
                  voltage, 0.00174
                   signal, 0.00172
               population, 0.00164
                     chip, 0.00160
                   cortex, 0.00158
                  private, 0.00148
                    ﬁring, 0.00148
                      Fig, 0.00140
                      ICA, 0.00130
                 cortical, 0.00129
--------------------------------------------------
Topic 7
               submodular, 0.00254
                    query, 0.00245
                     hash, 0.00235
                      EEG, 0.00211
                 movement, 0.00162
                      LSH, 0.00152
                      BCI, 0.00143
                   signal, 0.00143
                    motor, 0.00139
                    boost, 0.00138
               dictionary, 0.00131
                    prune, 0.00121
                   sparse, 0.00120
                   source, 0.00118
                     code, 0.00118
                   neuron, 0.00114
                 particle, 0.00113
                      arm, 0.00112
                   sensor, 0.00107
                 stimulus, 0.00106
               hypothesis, 0.00103
--------------------------------------------------
Topic 8
                    layer, 0.00468
              adversarial, 0.00426
                     deep, 0.00370
                    image, 0.00349
                  network, 0.00312
                    CIFAR, 0.00278
                   attack, 0.00275
            convolutional, 0.00268
            discriminator, 0.00245
                      GAN, 0.00215
                   neural, 0.00207
                   ResNet, 0.00204
                 preprint, 0.00200
                 training, 0.00200
              convolution, 0.00193
             architecture, 0.00189
                 ImageNet, 0.00186
                      CNN, 0.00184
                  dataset, 0.00181
                    MNIST, 0.00181
                generator, 0.00181
--------------------------------------------------
Topic 9
                   policy, 0.00489
                   regret, 0.00479
                   reward, 0.00330
                      arm, 0.00322
                   bandit, 0.00297
                   action, 0.00255
                     game, 0.00229
                   player, 0.00202
                   convex, 0.00179
                    round, 0.00168
                     bind, 0.00168
                   bounds, 0.00151
                    agent, 0.00150
                  Theorem, 0.00150
                       xt, 0.00140
                     loss, 0.00120
                      MDP, 0.00120
                   online, 0.00119
                 gradient, 0.00118
                    Lemma, 0.00117
                   oracle, 0.00115
--------------------------------------------------
Topic 10
                    agent, 0.00427
                   policy, 0.00379
                  network, 0.00273
                   reward, 0.00231
               trajectory, 0.00205
                   action, 0.00200
                     unit, 0.00187
              environment, 0.00183
                    state, 0.00173
                recurrent, 0.00172
            reinforcement, 0.00172
                    layer, 0.00165
                       RL, 0.00156
                      RNN, 0.00153
                     hide, 0.00153
                     LSTM, 0.00152
                     task, 0.00144
                   module, 0.00141
                   expert, 0.00136
                    robot, 0.00134
                   memory, 0.00134
'''




