import operator
import io

from gensim.parsing.preprocessing import strip_punctuation, strip_numeric, remove_stopwords, stem, strip_short
from gensim.parsing.preprocessing import preprocess_string
from gensim.corpora import Dictionary

'''
given a textual corpus, create vocabulary and convert textual corpus into a cleaned one
'''

class Dataset:
    '''
    self.id2word: gensim.corpora.dictionary
    self.text: list of list of words, sequential representation of document, used for obtaining word embedding
    '''
    def __init__(self, filepath, num_words):
        docs = []

        with io.open(filepath, 'r', encoding='utf-8') as f: # this may encounter utf-8 encoding error
            docs = [line for line in f.read().splitlines()]
        print(len(docs))

        # preprocessing and tokenization
        CUSTOMER_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_numeric, remove_stopwords, lambda x: strip_short(x, 3)]
        preprocess_func = lambda x: preprocess_string(x, CUSTOMER_FILTERS)
        docs = list(map(preprocess_func, docs))  # list of list of words, since gensim dictionary requires this format

        # generate dictionary
        self.id2word = Dictionary(docs)
        self.id2word.filter_extremes(no_below=3, no_above=0.25, keep_n=num_words)
        self.id2word.compactify()
        print('vocabulary size:', len(self.id2word))

        # generate sequence
        seq_func = lambda x: [w for w in x if w in self.id2word.token2id]
        self.text = list(map(seq_func, docs))
        self.text = [doc for doc in self.text if len(doc) > 0] # remove empty doc
        print('corpus size:', len(self.text))

    def save_data(self, bow_file, vocab_file):
        with io.open(bow_file, 'w', encoding='utf-8') as f:
            for doc in self.text:
                f.write(' '.join(doc) + '\n')
        vocab = [i[0] for i in sorted(self.id2word.token2id.items(), key=operator.itemgetter(1))]
        with io.open(vocab_file, 'w', encoding='utf-8') as f:
            for v in vocab:
                f.write(v + '\n')


# This Python code defines a class [`Dataset`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Ff%3A%2FStableLDA%2FStableLDA%2Fdataset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A11%2C%22character%22%3A6%7D%7D%5D%2C%22566fe3e8-a074-4903-8977-e4b6bfc45c80%22%5D "Go to definition") that processes a textual corpus to create a vocabulary and convert the corpus into a cleaned, tokenized format. Here's a detailed breakdown of its functionality:

# ### Purpose
# The main purpose of this code is to:
# 1. **Read a textual corpus from a file.**
# 2. **Preprocess and tokenize the text.**
# 3. **Create a dictionary (vocabulary) of words.**
# 4. **Convert the text into a sequence of tokens based on the dictionary.**
# 5. **Save the processed data and vocabulary to files.**

# ### Detailed Breakdown

# #### Imports
# ```python
# import operator
# import io
# from gensim.parsing.preprocessing import strip_punctuation, strip_numeric, remove_stopwords, stem, strip_short
# from gensim.parsing.preprocessing import preprocess_string
# from gensim.corpora import Dictionary
# ```
# - **operator**: Used for sorting the vocabulary.
# - **io**: Used for file operations.
# - **gensim.parsing.preprocessing**: Provides various text preprocessing functions.
# - **gensim.corpora.Dictionary**: Used to create a dictionary of words.

# #### Class Definition
# ```python
# class Dataset:
# ```
# Defines the [`Dataset`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Ff%3A%2FStableLDA%2FStableLDA%2Fdataset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A11%2C%22character%22%3A6%7D%7D%5D%2C%22566fe3e8-a074-4903-8977-e4b6bfc45c80%22%5D "Go to definition") class.

# #### Constructor
# ```python
# def __init__(self, filepath, num_words):
# ```
# - **filepath**: Path to the text file containing the corpus.
# - **num_words**: Maximum number of words to keep in the dictionary.

# #### Reading the File
# ```python
# with io.open(filepath, 'r', encoding='utf-8') as f:
#     docs = [line for line in f.read().splitlines()]
# print(len(docs))
# ```
# - Reads the file line by line and stores each line as a document in the [`docs`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Ff%3A%2FStableLDA%2FStableLDA%2Fdataset.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A17%2C%22character%22%3A8%7D%7D%5D%2C%22566fe3e8-a074-4903-8977-e4b6bfc45c80%22%5D "Go to definition") list.
# - Prints the number of documents read.

# #### Preprocessing and Tokenization
# ```python
# CUSTOMER_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_numeric, remove_stopwords, lambda x: strip_short(x, 3)]
# preprocess_func = lambda x: preprocess_string(x, CUSTOMER_FILTERS)
# docs = list(map(preprocess_func, docs))
# ```
# - **CUSTOMER_FILTERS**: A list of preprocessing functions to apply to each document.
# - **preprocess_func**: A lambda function that applies the preprocessing filters.
# - **docs**: The list of documents after preprocessing and tokenization.

# #### Dictionary Creation
# ```python
# self.id2word = Dictionary(docs)
# self.id2word.filter_extremes(no_below=3, no_above=0.25, keep_n=num_words)
# self.id2word.compactify()
# print('vocabulary size:', len(self.id2word))
# ```
# - **self.id2word**: Creates a dictionary from the tokenized documents.
# - **filter_extremes**: Filters out words that are too rare or too common.
# - **compactify**: Removes gaps in the word IDs.
# - Prints the size of the vocabulary.

# #### Sequence Generation
# ```python
# seq_func = lambda x: [w for w in x if w in self.id2word.token2id]
# self.text = list(map(seq_func, docs))
# self.text = [doc for doc in self.text if len(doc) > 0]
# print('corpus size:', len(self.text))
# ```
# - **seq_func**: A lambda function that keeps only the words present in the dictionary.
# - **self.text**: The list of documents after filtering out words not in the dictionary.
# - Removes empty documents.
# - Prints the size of the processed corpus.

# #### Saving Data
# ```python
# def save_data(self, bow_file, vocab_file):
#     with io.open(bow_file, 'w', encoding='utf-8') as f:
#         for doc in self.text:
#             f.write(' '.join(doc) + '\n')
#     vocab = [i[0] for i in sorted(self.id2word.token2id.items(), key=operator.itemgetter(1))]
#     with io.open(vocab_file, 'w', encoding='utf-8') as f:
#         for v in vocab:
#             f.write(v + '\n')
# ```
# - **save_data**: Saves the processed documents and vocabulary to files.
# - **bow_file**: File to save the bag-of-words representation of the documents.
# - **vocab_file**: File to save the vocabulary.

# ### Summary
# This code is useful for preparing textual data for natural language processing (NLP) tasks. It reads raw text, preprocesses it, creates a vocabulary, converts the text into a tokenized format, and saves the results for further use.