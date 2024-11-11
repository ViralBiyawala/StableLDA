
import io
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from statsmodels.discrete.discrete_model import Poisson
from patsy import dmatrices

# Read in original dataset, the stackexchange dataset
df = pd.read_csv('../data/stackexchange.csv', engine='python', on_bad_lines='skip')
print(df.head())

# Total number of documents
print(len(df.post.unique()) + len(df.answer.unique()))

# Read in data (the bow file and vocab file) for topic modeling.
texts = []
with io.open('../data/stackexchange.bow', 'r', encoding='utf-8') as f:
    texts = f.read().splitlines()

vocabs = []
with io.open('data/stackexchange.vocab', 'r', encoding='utf-8') as f:
    vocabs = f.read().splitlines()

# Using StableLDA to infer topic vectors.
from utils.python.stability import *
from utils.python.stablelda import StableLDA

# First run
bow_file = 'data/stackexchange.bow'
vocab_file = 'data/stackexchange.vocab'

num_topics = 50
num_words = 5000
alpha, beta, eta = 1, 0.01, 1000
epochs = 2
rand_seed = 42

# First model
output_dir = 'data/output/'
stablelda = StableLDA(num_topics, num_words, alpha, beta, eta, rand_seed, output_dir)
stablelda.train(bow_file, vocab_file, epochs)

docs, vocab, theta, phi = load_topic_model_results(bow_file, vocab_file,
                                                   output_dir+'theta.dat', output_dir+'phi.dat')
tm = TopicModel(num_topics, theta, phi, docs, vocab)

tm.print_top_n_words(10)

# Generate QASimilarity variable
df['post_idx'] = df['post_idx'].astype(int)
df['answer_idx'] = df['answer_idx'].astype(int)
df['stablelda_sim'] = df.apply(lambda x: 1 - cosine(theta[x['post_idx']], theta[x['answer_idx']]), axis=1)

# Run regression
y, X = dmatrices('AnswerHepfulness ~ stablelda_sim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.OLS(y, X)
ols_res = model.fit()
print(ols_res.summary())

y, X = dmatrices('dummy ~ stablelda_sim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.Logit(y, X)
logit_res = model.fit()
print(logit_res.summary())

results = summary_col([ols_res, logit_res], stars=True, float_format='%0.3f',
                      model_names=['OLS', 'Logit'],
                      info_dict={'Log-Likelihood': lambda x: "%#8.5g" % x.llf,
                                 'AIC': lambda x: "%#8.5g" % x.aic})
print(results)

# Second run
rand_seed = 24
stablelda = StableLDA(num_topics, num_words, alpha, beta, eta, rand_seed, output_dir)
stablelda.train(bow_file, vocab_file, epochs)

docs, vocab, theta, phi = load_topic_model_results(bow_file, vocab_file,
                                                   output_dir+'theta.dat', output_dir+'phi.dat')
tm = TopicModel(num_topics, theta, phi, docs, vocab)

df['stablelda_sim'] = df.apply(lambda x: 1 - cosine(theta[x['post_idx']], theta[x['answer_idx']]), axis=1)

# Linear regression
y, X = dmatrices('AnswerHepfulness ~ stablelda_sim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.OLS(y, X)
ols_res = model.fit()
print(ols_res.summary())

# Logit regression
y, X = dmatrices('dummy ~ stablelda_sim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.Logit(y, X)
logit_res = model.fit()
print(logit_res.summary())

results = summary_col([ols_res, logit_res], stars=True, float_format='%0.3f',
                      model_names=['OLS', 'Logit'],
                      info_dict={'Log-Likelihood': lambda x: "%#8.5g" % x.llf,
                                 'AIC': lambda x: "%#8.5g" % x.aic})
print(results)

# Using LDA to infer topic vectors.
import pickle
import gensim

# Load data
gensimcorpus = pickle.load(open('data/stackexchange.gaming.corpus.gensim', 'rb'))
id2word = pickle.load(open('data/stackexchange.gaming.id2word.gensim', 'rb'))

# LDA first run
lda_model = gensim.models.LdaMulticore(gensimcorpus, num_topics=num_topics, alpha='symmetric', id2word=id2word, passes=10)

lda_theta = []
for bow in gensimcorpus:
    prob = [i[1] for i in lda_model.get_document_topics(bow, minimum_probability=0)]
    lda_theta.append(prob)
df['lda_sim'] = df.apply(lambda x: 1 - cosine(lda_theta[x['post_idx']], lda_theta[x['answer_idx']]), axis=1)

# Linear regression
y, X = dmatrices('AnswerHepfulness ~ lda_sim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.OLS(y, X)
ols_res = model.fit()
print(ols_res.summary())

# Logit regression
y, X = dmatrices('dummy ~ lda_sim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.Logit(y, X)
logit_res = model.fit()
print(logit_res.summary())

# LDA second run
lda_model = gensim.models.LdaMulticore(gensimcorpus, num_topics=num_topics, alpha='symmetric', id2word=id2word, passes=10)

lda_theta = []
for bow in gensimcorpus:
    prob = [i[1] for i in lda_model.get_document_topics(bow, minimum_probability=0)]
    lda_theta.append(prob)
df['lda_sim'] = df.apply(lambda x: 1 - cosine(lda_theta[x['post_idx']], lda_theta[x['answer_idx']]), axis=1)

# Linear regression
y, X = dmatrices('AnswerHepfulness ~ lda_sim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.OLS(y, X)
ols_res = model.fit()
print(ols_res.summary())

# Logit regression
y, X = dmatrices('dummy ~ lda_sim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.Logit(y, X)
logit_res = model.fit()
print(logit_res.summary())

# Robustness check using TF-IDF
vectorizer = TfidfVectorizer(vocabulary=vocabs)
X = vectorizer.fit_transform(texts)

tfidfsim = []
for idx, row in df.iterrows():
    post_tfidf = X[row.post_idx].todense()
    answer_tfidf = X[row.answer_idx].todense()
    tfidfsim.append(1 - cosine(post_tfidf, answer_tfidf))
df['tfidfsim'] = pd.Series(list(tfidfsim))

# Run regression to examine the relationship between QA similarity and answer helpfulness
y, X = dmatrices('AnswerHepfulness ~ tfidfsim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.OLS(y, X)
ols_res = model.fit()
print(ols_res.summary())

y, X = dmatrices('dummy ~ tfidfsim + Sequence + QuestionHelpfulness + logwords', data=df, return_type='dataframe')
model = sm.Logit(y, X)
logit_res = model.fit()
print(logit_res.summary())

results = summary_col([ols_res, logit_res], stars=True, float_format='%0.3f',
                      model_names=['OLS', 'Logit'],
                      info_dict={'Log-Likelihood': lambda x: "%#8.5g" % x.llf,
                                 'AIC': lambda x: "%#8.5g" % x.aic})
print(results)