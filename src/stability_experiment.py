from utils.python.stablelda import StableLDA
from utils.python.stability import load_topic_model_results, model_alignment, theta_stability, doc_stability, phi_stability, topwords_stability, TopicModel
import os

bow_file = '../data/stackexchange.bow'
vocab_file = '../data/stackexchange.vocab'

num_topics = 10
num_words = 5000
alpha, beta, eta = 1, 0.01, 1000
epochs = 5

os.system('cd .. && mingw32-make')  # navigate to the parent folder and run mingw32-make

output_dir = 'output/model1/'
rand_seed = 42

stablelda = StableLDA(num_topics, num_words, alpha, beta, eta, rand_seed, output_dir)
stablelda.train(bow_file, vocab_file, epochs)

docs, vocab, theta1, phi1 = load_topic_model_results(bow_file, vocab_file,
                                                     output_dir + 'theta.dat', output_dir + 'phi.dat')
tm1 = TopicModel(num_topics, theta1, phi1, docs, vocab)

tm1.print_top_n_words(10)
print('----------------------------------------------')

output_dir = 'output/model2/'
rand_seed = 24

stablelda = StableLDA(num_topics, num_words, alpha, beta, eta, rand_seed, output_dir)
stablelda.train(bow_file, vocab_file, epochs)

docs, vocab, theta2, phi2 = load_topic_model_results(bow_file, vocab_file,
                                                     output_dir + 'theta.dat', output_dir + 'phi.dat')
tm2 = TopicModel(num_topics, theta2, phi2, docs, vocab)

tm2.print_top_n_words(10)
print('----------------------------------------------')

alignment = model_alignment(tm1, tm2)

print('doc topic stability:', theta_stability(tm1, tm2, alignment))
print('doc label stability:', doc_stability(tm1, tm2, alignment))
print('topic word stability:', phi_stability(tm1, tm2, alignment))
print('top 10 word stability:', topwords_stability(tm1, tm2, alignment))
