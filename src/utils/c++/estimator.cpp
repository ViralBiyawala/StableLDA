// This file defines the Estimator class, which is used for estimating the parameters of a topic model using a Dirichlet tree structure. 
// The class includes methods for reading vocabulary and data files, building the Dirichlet tree, performing Gibbs sampling to estimate topic distributions, 
// and calculating and saving the resulting topic-word and document-topic distributions.
//
// The main components of the Estimator class are:
// 1. Reading Vocabulary and Data Files: Methods to read and parse the vocabulary and data files which contain the words and documents respectively.
// 2. Building the Dirichlet Tree: Methods to construct the Dirichlet tree structure which is used to model the hierarchical relationships between topics.
// 3. Gibbs Sampling: Methods to perform Gibbs sampling, a Markov Chain Monte Carlo (MCMC) algorithm, to estimate the topic distributions for the words in the documents.
// 4. Calculating Distributions: Methods to calculate the topic-word distributions (probability of words given topics) and document-topic distributions (probability of topics given documents).
// 5. Saving Results: Methods to save the estimated topic-word and document-topic distributions to files for further analysis or use.
