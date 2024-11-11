
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <locale>

class Dataset {
public:
    Dataset(const std::string& filepath, int num_words);
    void save_data(const std::string& bow_file, const std::string& vocab_file);

private:
    std::unordered_map<std::string, int> id2word;
    std::vector<std::vector<std::string>> text;
    void preprocess(std::vector<std::string>& docs);
    std::vector<std::string> tokenize(const std::string& str);
    std::string to_lower(const std::string& str);
    std::string remove_punctuation(const std::string& str);
    std::string remove_numeric(const std::string& str);
    std::string remove_stopwords(const std::string& str);
    std::string strip_short(const std::string& str, size_t min_length);
};

Dataset::Dataset(const std::string& filepath, int num_words) {
    std::vector<std::string> docs;
    std::ifstream file(filepath);
    std::string line;
    while (std::getline(file, line)) {
        docs.push_back(line);
    }
    std::cout << docs.size() << std::endl;

    preprocess(docs);

    // Generate dictionary
    std::unordered_map<std::string, int> word_freq;
    for (const auto& doc : docs) {
        for (const auto& word : tokenize(doc)) {
            word_freq[word]++;
        }
    }

    // Filter extremes and keep top num_words
    std::vector<std::pair<std::string, int>> sorted_words(word_freq.begin(), word_freq.end());
    std::sort(sorted_words.begin(), sorted_words.end(), [](const auto& a, const auto& b) {
        return b.second < a.second;
    });

    for (size_t i = 0; i < std::min(num_words, static_cast<int>(sorted_words.size())); ++i) {
        id2word[sorted_words[i].first] = i;
    }
    std::cout << "vocabulary size: " << id2word.size() << std::endl;

    // Generate sequence
    for (const auto& doc : docs) {
        std::vector<std::string> seq;
        for (const auto& word : tokenize(doc)) {
            if (id2word.find(word) != id2word.end()) {
                seq.push_back(word);
            }
        }
        if (!seq.empty()) {
            text.push_back(seq);
        }
    }
    std::cout << "corpus size: " << text.size() << std::endl;
}

void Dataset::preprocess(std::vector<std::string>& docs) {
    for (auto& doc : docs) {
        doc = to_lower(doc);
        doc = remove_punctuation(doc);
        doc = remove_numeric(doc);
        doc = remove_stopwords(doc);
        doc = strip_short(doc, 3);
    }
}

std::vector<std::string> Dataset::tokenize(const std::string& str) {
    std::istringstream iss(str);
    return std::vector<std::string>{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
}

std::string Dataset::to_lower(const std::string& str) {
    std::string result;
    std::transform(str.begin(), str.end(), std::back_inserter(result), ::tolower);
    return result;
}

std::string Dataset::remove_punctuation(const std::string& str) {
    std::string result;
    std::remove_copy_if(str.begin(), str.end(), std::back_inserter(result), ::ispunct);
    return result;
}

std::string Dataset::remove_numeric(const std::string& str) {
    std::string result;
    std::remove_copy_if(str.begin(), str.end(), std::back_inserter(result), ::isdigit);
    return result;
}

std::string Dataset::remove_stopwords(const std::string& str) {
    // Implement stopword removal based on a predefined list of stopwords
    // For simplicity, this function is left as a placeholder
    return str;
}

std::string Dataset::strip_short(const std::string& str, size_t min_length) {
    std::istringstream iss(str);
    std::string word;
    std::string result;
    while (iss >> word) {
        if (word.length() >= min_length) {
            result += word + " ";
        }
    }
    return result;
}

void Dataset::save_data(const std::string& bow_file, const std::string& vocab_file) {
    std::ofstream bow_out(bow_file);
    for (const auto& doc : text) {
        for (const auto& word : doc) {
            bow_out << word << " ";
        }
        bow_out << "\n";
    }

    std::ofstream vocab_out(vocab_file);
    for (const auto& pair : id2word) {
        vocab_out << pair.first << "\n";
    }
}