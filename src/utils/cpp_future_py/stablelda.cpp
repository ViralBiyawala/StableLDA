
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>

class StableLDA {
public:
    StableLDA(int num_topics, int num_words, double alpha, double beta, double eta, int rand_seed, const std::string& output_dir, const std::string& embed_method = "cbow");
    void load_data(const std::string& bow_file, const std::string& vocab_file);
    void train(const std::string& bow_file, const std::string& vocab_file, int epochs);
    void save_intermediate();
    void init_word_cluster(const std::string& embed_method);
    void initialize();
    void inference(int epochs);

private:
    int num_topics;
    int num_words;
    int num_cluster;
    double alpha;
    double beta;
    double eta;
    int rand_seed;
    std::string embed_method;
    std::string output_dir;
    std::vector<std::vector<int>> text;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> vocab2id;
    std::vector<std::vector<int>> bow;
    std::string bow_file;
    std::string vocab_file;
    std::string cluster_file;
    std::string sample_file;
    std::vector<std::vector<int>> zsampes;
    std::vector<std::vector<std::string>> topical_clusters;
    std::unordered_map<int, std::unordered_map<int, int>> word_topic;
    std::unordered_map<int, std::unordered_map<int, float>> topic_word;
    std::vector<std::unordered_map<int, float>> doc_topic;
    std::vector<std::unordered_map<int, int>> doc_assignments;
    std::unordered_map<int, int> word2part;
    // ...other member variables...
};

StableLDA::StableLDA(int num_topics, int num_words, double alpha, double beta, double eta, int rand_seed, const std::string& output_dir, const std::string& embed_method)
    : num_topics(num_topics), num_words(num_words), alpha(alpha), beta(beta), eta(eta), rand_seed(rand_seed), embed_method(embed_method), output_dir(output_dir) {
    std::cout << "--------running Stable LDA model----------" << std::endl;
    num_cluster = num_topics + 10;
    struct stat info;
    if (stat(output_dir.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
        mkdir(output_dir.c_str(), 0777);
    }
}

void StableLDA::load_data(const std::string& bow_file, const std::string& vocab_file) {
    std::cout << "--------- loading data ----------------" << std::endl;
    // ...existing code...
}

void StableLDA::train(const std::string& bow_file, const std::string& vocab_file, int epochs) {
    load_data(bow_file, vocab_file);
    init_word_cluster(embed_method);
    initialize();
    save_intermediate();
    inference(epochs);
}

void StableLDA::save_intermediate() {
    std::ofstream sample_out(sample_file);
    for (const auto& sample : zsampes) {
        for (const auto& z : sample) {
            sample_out << z << " ";
        }
        sample_out << "\n";
    }
    sample_out.close();

    std::ofstream cluster_out(cluster_file);
    for (const auto& cluster : topical_clusters) {
        for (const auto& word : cluster) {
            cluster_out << word << ",";
        }
        cluster_out << "\n";
    }
    cluster_out.close();
}

void StableLDA::init_word_cluster(const std::string& embed_method) {
    // ...existing code...
}

void StableLDA::initialize() {
    // ...existing code...
}

void StableLDA::inference(int epochs) {
    std::cout << "--------- inference ----------------" << std::endl;
    std::string cmd = "train"; // windows
    // cmd = "./train"; // linux
    cmd += " -f " + bow_file;
    cmd += " -v " + vocab_file;
    cmd += " -c " + cluster_file;
    cmd += " -z " + sample_file;
    cmd += " -t " + std::to_string(num_topics);
    cmd += " -w " + std::to_string(num_words);
    cmd += " -a " + std::to_string(alpha);
    cmd += " -b " + std::to_string(beta);
    cmd += " -e " + std::to_string(eta);
    cmd += " -n " + std::to_string(epochs);
    cmd += " -r " + std::to_string(rand_seed);
    cmd += " -o " + output_dir;

    std::cout << cmd << std::endl;
    system(cmd.c_str());
}