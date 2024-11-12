# Stable LDA: Extracting Actionable Insights from Text Data

### Introduction
Stable LDA is a topic modeling technique that produces stable model estimations for consistent regression analysis. This repository demonstrates Stable LDA for both topic modeling (data exploration) and regressions (variable generation). The core scripts include:
1. `stability_experiment.py`: Uses Stable LDA for topic modeling and stability validation.
2. `stackexchange_empirical.py`: Applies Stable LDA to the StackExchange dataset with an LDA benchmark.
3. `stackexchange_topic_modeling.py`: Demonstrates Stable LDA for topic modeling.

### Environment Requirements
1. **Python 3.10**
2. **Packages**: 
   - `gensim==4.3.3`
   - `scipy==1.13.1`
   - `scikit-learn==1.5.2`
3. **Compiler**:
   - **Linux**: `gcc==9.4.0`
   - **Windows**: `mingw32-make` (GNU Make 4.3)

### Usage Guide
#### For Windows Users:
1. In `Makefile`: Comment line 14 and uncomment line 13.
2. In `stablelda.py`: Comment line 313 and uncomment line 312.
3. Run:
   ``` 
   mingw32-make 
   ```

#### For Linux Users:
1. In `Makefile`: Comment line 13 and uncomment line 14.
2. In `stablelda.py`: Comment line 312 and uncomment line 313.
3. Run:
   ``` 
   make 
   ```

### Code Overview

This repository includes Python scripts that implement and test the StableLDA model for stable topic modeling.

#### stablelda.py
Implements the `StableLDA` class, defining methods for:
- **Data Loading** (`load_data`): Loads bag-of-words (BOW) data and vocabulary.
- **Training** (`train`): Runs training with intermediate data saving.
- **Cluster Initialization** (`init_word_cluster`): Uses Word2Vec or FastText embeddings with KMeans clustering.
- **Inference** (`inference`): Executes inference by calling an external C++ program.

#### stability.py
Implements `TopicModel` class and functions to assess topic model stability:
- **Model Alignment** (`model_alignment`): Aligns two models using a cost matrix.
- **Stability Metrics**: Includes document-topic distribution (`theta_stability`), document label stability (`doc_stability`), and word-topic distribution stability (`phi_stability`).

#### dataset.py
Defines `Dataset` class for text preprocessing:
- **Corpus Preprocessing** (`__init__`): Loads, tokenizes, and creates dictionary from text data.
- **Data Saving** (`save_data`): Saves BOW and vocabulary.

### Running the Code

1. **Dataset Preparation**
   - Process text corpus to create BOW and vocabulary.
   ```python
   dataset = Dataset(filepath='data/stackexchange.csv', num_words=10000)
   dataset.save_data(bow_file='data/stackexchange.bow', vocab_file='data/stackexchange.vocab')
   ```

2. **Training the Model**
   - Initialize and train `StableLDA`.
   ```python
   stable_lda = StableLDA(num_topics=10, num_words=10000, alpha=0.1, beta=0.01, eta=0.01, rand_seed=42, output_dir='src/output/model1/')
   stable_lda.train(bow_file='data/stackexchange.bow', vocab_file='data/stackexchange.vocab', epochs=100)
   ```

3. **Stability Evaluation**
   - Load models and evaluate stability.
   ```python
   docs, vocab, theta, phi = load_topic_model_results(doc_path='data/stackexchange.bow', vocab_path='data/stackexchange.vocab', theta_path='src/output/model1/theta.dat', phi_path='src/output/model1/phi.dat')
   tm1 = TopicModel(num_topics=10, theta=theta, beta=phi, bows=docs, vocab=vocab)
   tm2 = TopicModel(num_topics=10, theta=theta, beta=phi, bows=docs, vocab=vocab)
   alignment = model_alignment(tm1, tm2)
   stability = theta_stability(tm1, tm2, alignment)
   print('Theta Stability:', stability)
   ```

### C++ Code Overview

#### Data Flow
1. **Input Data**: Includes `data/stackexchange.bow`, `data/stackexchange.vocab`, `src/output/model1/cluster.dat`, `src/output/model1/z.dat`.
2. **Tree Construction** (`build_tree`): Builds a Dirichlet hierarchy for topic distribution.
3. **Gibbs Sampling** (`estimate`): Estimates topic distributions via MCMC sampling.
4. **Distributions Calculation** (`calc_theta` and `calc_phi`): Produces document-topic (`theta`) and topic-word (`phi`) distributions.
5. **Result Saving**: Saves estimated distributions.