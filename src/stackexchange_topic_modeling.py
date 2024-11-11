
import io
import pandas as pd
import gensim
from utils.python.stability import *
from utils.python.stablelda import StableLDA

# train a topic model of 25 topics on the stackexchange dataset using Stable LDA
bow_file = '../data/stackexchange.bow'
vocab_file = '../data/stackexchange.vocab'

num_topics = 25
num_words = 5000
alpha, beta, eta = 1, 0.01, 1000
epochs = 5
rand_seed = 42
output_dir = 'data/output/'

stablelda = StableLDA(num_topics, num_words, alpha, beta, eta, rand_seed, output_dir)
stablelda.train(bow_file, vocab_file, epochs)

docs, vocab, theta, phi = load_topic_model_results(bow_file, vocab_file,
                                                   output_dir+'theta.dat', output_dir+'phi.dat')
tm = TopicModel(num_topics, theta, phi, docs, vocab)

# tm.print_top_n_words(10)

# compute model perplexity
compute_perlexity(docs, theta, phi)

# compute model coherence.
# we use Gensim's coherence method. We need to prepare gensim_bow, and id2word
topics = tm.get_top_n_words(10)

# read in raw text data -- used for windows-based topic coherence measure
with io.open(bow_file, 'r', encoding='utf-8') as f:
    texts = [line.split() for line in f.read().splitlines()]

# prepare gensim_bow and id2word
id2word = gensim.corpora.Dictionary(texts)
gensim_bow = [id2word.doc2bow(text) for text in texts]

print('topic coherence c_uci', compute_coherence(gensim_bow, texts, id2word, topics, coherence_score='c_uci'))
print('topic coherence c_v', compute_coherence(gensim_bow, texts, id2word, topics, coherence_score='c_v'))

tm.print_top_n_words(10)