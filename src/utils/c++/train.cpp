#include <iostream>
#include <getopt.h>
#include "estimator.h"
#include "utility.h"

using namespace std;
using namespace utils;

int main(int argc, char *argv[]) {

	int opt;
	string data_file;
	string cluster_file;
	string z_file;
	string vocab_file;
	string output_path;
	int num_words, num_topics;
	double alpha, beta, eta;
	int rand_seed;
	int epochs;

	const char *optstring = "f:v:c:z:t:w:a:b:e:n:r:o:";

	while( (opt = getopt(argc, argv, optstring)) != -1){

		switch (opt){
			case 'f':
				data_file = optarg;
				break;
			case 'v':
				vocab_file = optarg;
				break;
			case 'c':
				cluster_file = optarg;
				break;
			case 'z':
				z_file = optarg;
				break;
			case 't':
				num_topics = atoi(optarg);
				break;
			case 'w':
				num_words = atoi(optarg);
				break;
			case 'a':
				alpha = atof(optarg);
				break;
			case 'b':
				beta = atof(optarg);
				break;
			case 'e':
				eta = atoi(optarg);
				break;
			case 'n':
				epochs = atoi(optarg);
				break;
			case 'r':
				rand_seed = atoi(optarg);
				break;
			case 'o':
				output_path = optarg;
				break;
			default:
				cerr <<"unknown option: " << char(optopt) << endl;
				return -1;
		}

	}

    Estimator est(alpha, beta, eta, num_topics, num_words, rand_seed);
	cout << "loading data - train.cpp" << endl;
    est.load_data(data_file, z_file, cluster_file, vocab_file);
    est.estimate(epochs);

    est.save(output_path);

	return 0;
}

/*
 * This file is the main entry point for training a topic model using the Estimator class.
 * It parses command-line arguments to set various parameters such as the data file, vocabulary file,
 * cluster file, number of topics, number of words, alpha, beta, eta, number of epochs, random seed, 
 * and output path. After parsing the arguments, it initializes an Estimator object with these parameters.
 * The Estimator object then loads the data, performs the estimation process for the specified number of epochs,
 * and finally saves the results to the specified output path.
 */
