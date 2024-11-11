
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <numeric>
#include <iterator>
#include <functional>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using namespace std;
using namespace Eigen;

class TopicModel {
public:
    int num_topics;
    MatrixXd theta;
    MatrixXd beta;
    vector<vector<int>> bows;
    vector<string> vocab;
    int num_docs;
    vector<vector<string>> topnwords;
    vector<int> doc_labels;
    vector<set<int>> clusters;

    TopicModel(int num_topics, MatrixXd theta, MatrixXd beta, vector<vector<int>> bows, vector<string> vocab)
        : num_topics(num_topics), theta(theta), beta(beta), bows(bows), vocab(vocab) {
        num_docs = bows.size();
        get_top_n_words();
        get_doc_labels();
        get_doc_clusters();
    }

    vector<vector<string>> get_top_n_words(int n = 10) {
        topnwords.clear();
        for (int k = 0; k < beta.rows(); ++k) {
            vector<pair<double, int>> prob_idx;
            for (int j = 0; j < beta.cols(); ++j) {
                prob_idx.push_back(make_pair(beta(k, j), j));
            }
            sort(prob_idx.rbegin(), prob_idx.rend());
            vector<string> topnwords_k;
            for (int j = 0; j < n; ++j) {
                topnwords_k.push_back(vocab[prob_idx[j].second]);
            }
            topnwords.push_back(topnwords_k);
        }
        return topnwords;
    }

    void print_top_n_words(int n = 10) {
        for (const auto& words : topnwords) {
            copy(words.begin(), words.end(), ostream_iterator<string>(cout, " "));
            cout << endl;
        }
    }

    void get_doc_labels() {
        doc_labels.clear();
        for (int i = 0; i < theta.rows(); ++i) {
            int label = distance(theta.row(i).data(), max_element(theta.row(i).data(), theta.row(i).data() + theta.cols()));
            doc_labels.push_back(label);
        }
    }

    void get_doc_clusters() {
        clusters.clear();
        clusters.resize(num_topics);
        for (int i = 0; i < theta.rows(); ++i) {
            int max_topic = distance(theta.row(i).data(), max_element(theta.row(i).data(), theta.row(i).data() + theta.cols()));
            clusters[max_topic].insert(i);
        }
    }
};

MatrixXd computeMatrix(TopicModel& tm1, TopicModel& tm2) {
    if (tm1.num_topics != tm2.num_topics) {
        throw invalid_argument("two topic models have different topics");
    }

    MatrixXd matrix = MatrixXd::Zero(tm1.num_topics, tm1.num_topics);
    for (int i = 0; i < tm1.num_topics; ++i) {
        for (int j = 0; j < tm1.num_topics; ++j) {
            matrix(i, j) = tm1.clusters[i].size() + tm2.clusters[j].size() - 2 * set_intersection(tm1.clusters[i].begin(), tm1.clusters[i].end(), tm2.clusters[j].begin(), tm2.clusters[j].end(), ostream_iterator<int>(cout, " ")).size();
        }
    }
    return matrix;
}

unordered_map<int, int> model_alignment(TopicModel& tm1, TopicModel& tm2) {
    MatrixXd cost_matrix = computeMatrix(tm1, tm2);
    // Implement linear_sum_assignment equivalent in C++
    // ...
    unordered_map<int, int> alignment_dict;
    // Fill alignment_dict based on the result of linear_sum_assignment
    // ...
    return alignment_dict;
}

TopicModel align_a_tm(TopicModel& tm, unordered_map<int, int>& alignment) {
    MatrixXd old_theta = tm.theta;
    MatrixXd old_beta = tm.beta;
    MatrixXd new_theta = MatrixXd::Zero(old_theta.rows(), old_theta.cols());
    MatrixXd new_beta = MatrixXd::Zero(old_beta.rows(), old_beta.cols());

    for (int k = 0; k < tm.num_topics; ++k) {
        new_theta.col(alignment[k]) = old_theta.col(k);
        new_beta.row(alignment[k]) = old_beta.row(k);
    }

    return TopicModel(tm.num_topics, new_theta, new_beta, tm.bows, tm.vocab);
}

double theta_stability(TopicModel& tm1, TopicModel& tm2, unordered_map<int, int>& alignment) {
    vector<double> l1_distances;
    for (int i = 0; i < tm1.num_docs; ++i) {
        double dist = 0.0;
        for (int k = 0; k < tm1.num_topics; ++k) {
            dist += abs(tm1.theta(i, k) - tm2.theta(i, alignment[k]));
        }
        l1_distances.push_back(1 - 0.5 * dist);
    }
    return accumulate(l1_distances.begin(), l1_distances.end(), 0.0) / l1_distances.size();
}

double doc_stability(TopicModel& tm1, TopicModel& tm2, unordered_map<int, int>& alignment) {
    vector<bool> index_match_bool;
    for (int i = 0; i < tm1.num_docs; ++i) {
        index_match_bool.push_back(tm1.doc_labels[i] == alignment[tm2.doc_labels[i]]);
    }
    return accumulate(index_match_bool.begin(), index_match_bool.end(), 0.0) / index_match_bool.size();
}

double phi_stability(TopicModel& tm1, TopicModel& tm2, unordered_map<int, int>& alignment) {
    vector<double> l1_distances;
    for (int k = 0; k < tm1.num_topics; ++k) {
        double dist = 0.0;
        for (int j = 0; j < tm1.beta.cols(); ++j) {
            dist += abs(tm1.beta(k, j) - tm2.beta(alignment[k], j));
        }
        l1_distances.push_back(1 - 0.5 * dist);
    }
    return accumulate(l1_distances.begin(), l1_distances.end(), 0.0) / l1_distances.size();
}

double topwords_stability(TopicModel& tm1, TopicModel& tm2, unordered_map<int, int>& alignment) {
    vector<double> similarity;
    for (int k = 0; k < tm1.num_topics; ++k) {
        set<string> set1(tm1.topnwords[alignment[k]].begin(), tm1.topnwords[alignment[k]].end());
        set<string> set2(tm2.topnwords[k].begin(), tm2.topnwords[k].end());
        vector<string> intersection;
        set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), back_inserter(intersection));
        similarity.push_back(static_cast<double>(intersection.size()) / 10.0);
    }
    return accumulate(similarity.begin(), similarity.end(), 0.0) / similarity.size();
}

double compute_perplexity(vector<vector<int>>& bow, MatrixXd& theta, MatrixXd& phi) {
    cout << "compute likelihood" << endl;
    double loglikelihood = 0.0;
    int wordcount = 0;
    for (int i = 0; i < bow.size(); ++i) {
        VectorXd doc_topic = theta.row(i);
        for (int w : bow[i]) {
            double pr = 0.0;
            wordcount++;
            for (int k = 0; k < doc_topic.size(); ++k) {
                pr += doc_topic[k] * phi(k, w);
            }
            loglikelihood += log(pr);
        }
    }
    cout << "likelihood: " << loglikelihood << endl;
    cout << "perplexity: " << exp(-loglikelihood / wordcount) << endl;
    return exp(-loglikelihood / wordcount);
}

double compute_coherence(vector<vector<int>>& gensim_bow, vector<vector<string>>& text, unordered_map<int, string>& id2word, vector<vector<int>>& topics, string coherence_score = "c_npmi") {
    // Implement coherence model calculation
    // ...
    return 0.0;
}

tuple<vector<vector<int>>, vector<string>, MatrixXd, MatrixXd> load_topic_model_results(string doc_path, string vocab_path, string theta_path, string phi_path) {
    vector<vector<int>> docs;
    vector<string> vocab;
    MatrixXd theta;
    MatrixXd phi;
    unordered_map<string, int> vocab2id;

    ifstream vocab_file(vocab_path);
    string line;
    while (getline(vocab_file, line)) {
        vocab.push_back(line);
        vocab2id[line] = vocab.size() - 1;
    }

    ifstream doc_file(doc_path);
    while (getline(doc_file, line)) {
        istringstream iss(line);
        vector<int> doc;
        string word;
        while (iss >> word) {
            doc.push_back(vocab2id[word]);
        }
        docs.push_back(doc);
    }

    ifstream theta_file(theta_path);
    vector<vector<double>> theta_vec;
    while (getline(theta_file, line)) {
        istringstream iss(line);
        vector<double> row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        theta_vec.push_back(row);
    }
    theta = MatrixXd(theta_vec.size(), theta_vec[0].size());
    for (int i = 0; i < theta_vec.size(); ++i) {
        for (int j = 0; j < theta_vec[i].size(); ++j) {
            theta(i, j) = theta_vec[i][j];
        }
    }

    ifstream phi_file(phi_path);
    vector<vector<double>> phi_vec;
    while (getline(phi_file, line)) {
        istringstream iss(line);
        vector<double> row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        phi_vec.push_back(row);
    }
    phi = MatrixXd(phi_vec.size(), phi_vec[0].size());
    for (int i = 0; i < phi_vec.size(); ++i) {
        for (int j = 0; j < phi_vec[i].size(); ++j) {
            phi(i, j) = phi_vec[i][j];
        }
    }

    return make_tuple(docs, vocab, theta, phi);
}