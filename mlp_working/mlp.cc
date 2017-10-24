/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <cstdio>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <libconfig.h++>
#include <random>
#include <string>
#include "mlp.h"

using namespace std;
using namespace libconfig;



mlp_t::mlp_t() :
    neuron(NULL),
    width(0), length(0),
    require_training(false),
    test_img_set(NULL),
    train_img_set(NULL),
    test_label_set(NULL),
    train_label_set(NULL),
	answer_set(NULL) {
}

mlp_t::~mlp_t() {
    if(weights) {
        for(unsigned i = 0; i < total_layers_index; i++){
            delete [] weights[i];
        }
    }
	if(delta) {
		for(unsigned i = 0; i < total_layers_index; i++) {
			delete [] delta[i];
		}
	}
    delete [] weights;
    delete [] test_img_set;
    delete [] test_label_set;
    delete [] train_img_set;
    delete [] train_label_set;
	delete [] answer_set;
    delete [] num_neurons_per_layer;
    delete [] delta;
}

void mlp_t::initialize(string m_config_file_name) {

    // User libconfig to parse a configuration file.
    config_file_name = m_config_file_name;
    Config mlp_config;
    mlp_config.readFile(config_file_name.c_str());

    try {
        test_img_file_name = mlp_config.lookup("test_img").c_str();
        test_label_file_name = mlp_config.lookup("test_label").c_str();
        if(mlp_config.exists("weight")) {
            weight_file_name = mlp_config.lookup("weight").c_str();
            if(mlp_config.exists("train_img") ||
               mlp_config.exists("train_label")) {
                cout << "Warning: train_img and train_label will be ignored" << endl;
            }
            require_training = false;
        }
        else {
            train_img_file_name = mlp_config.lookup("train_img").c_str();
            train_label_file_name = mlp_config.lookup("train_label").c_str();
            if(mlp_config.exists("weight")) {
                cout << "Warning: pre-trained weight will be ignored" << endl;
            }
            require_training = true;
        }

        // Load hidden layer and image size array settings.
        Setting &s_num_neurons_in_hidden_layer = mlp_config.lookup("num_neurons_in_hidden_layer");
        Setting &s_image_size = mlp_config.lookup("image_size");
        Setting &s_num_neurons_in_output_layer = mlp_config.lookup("num_neurons_in_output_layer");

        // +2 means 1 for input and another 1 for output layer
        num_layers = s_num_neurons_in_hidden_layer.getLength()+2;
		total_layers_index = num_layers-1;

        // Load the # of neurons in the input layer.
        if(s_image_size.getLength() != 2) {
            cerr << "image_size must be defined [width, length]" << endl;
            exit(1);
        }
       
        // Setting each layer
        num_neurons_per_layer = new unsigned[num_layers];
        num_neurons_in_input_layer = unsigned(s_image_size[0]) * unsigned(s_image_size[1]);
       
        // Load the # of neurons in input layer.
        num_neurons_per_layer[0] = num_neurons_in_input_layer;
        
        // Load the number of neurons in each hidden layer.
        for(int i = 1; i <= s_num_neurons_in_hidden_layer.getLength(); i++) {
            num_neurons_per_layer[i] = unsigned(s_num_neurons_in_hidden_layer[i-1]);
        }

        // Load the # of neurons in output layer.
        num_neurons_per_layer[total_layers_index] = unsigned(s_num_neurons_in_output_layer);
		
		// Set Neuron
		neuron = new double*[num_layers];
		for(unsigned i = 0; i < num_layers; i++) {
			if(i == total_layers_index) {
				neuron[i] = new double[num_neurons_per_layer[i]];
			}
			else {
				neuron[i] = new double[num_neurons_per_layer[i]+1];
			}
		}
		for(unsigned i = 0; i < total_layers_index; i++) {
			neuron[i][num_neurons_per_layer[i]] = 0.5;
		}


        // Load the number of training and test set.
        test_set_size = unsigned(mlp_config.lookup("test_set_size"));
        train_set_size = unsigned(mlp_config.lookup("train_set_size"));

        // Load the Learning Rate.
        learning_rate = double(mlp_config.lookup("learning_rate"));
       
        // Setting training set into label and value.
        train_label_set = new double[train_set_size];
        train_img_set = new double[train_set_size * num_neurons_in_input_layer];
		answer_set = new double[num_neurons_per_layer[total_layers_index]];

        // Setting test set into label and value.
        test_label_set = new double[test_set_size];
        test_img_set = new double[test_set_size * num_neurons_in_input_layer];

        weights = new double*[total_layers_index];
        for(unsigned i = 0; i < total_layers_index; i++) {
			weights[i] = new double[(num_neurons_per_layer[i]+1)*num_neurons_per_layer[i+1]];
		}
       
        // Set values into weights.
        if(!weight_file_name.size()) { init_weights();
		cout << "init_weights" << endl;
		}
        else { load_weights(); 
		cout << "load_weights" << endl;
		} 

        // Setting delta
        delta = new double*[total_layers_index];
        for(unsigned i = 1; i < num_layers; i++) {
            delta[i-1] = new double[num_neurons_per_layer[i]];
        }

    }
    catch(SettingNotFoundException e) {
        cout << "Error: " << e.getPath() << " is not defined in "
             << config_file_name << endl;
    }
    catch(SettingTypeException e) {
        cout << "Error: " << e.getPath() << " has incorrect type in "
             << config_file_name << endl;
    }
    catch(FileIOException e) {
        cout << "Error: " << config_file_name << " does not exist" << endl;
    }
    catch(ParseException e) {
        cout << "Error: Failed to parse line # " << e.getLine()
             << " in " << config_file_name << endl;
    }
}

// Read test image file
void mlp_t::read_test_img_file() {
    fstream file_stream;
    file_stream.open(test_img_file_name.c_str(), fstream::in|fstream::binary);
     
    if(!file_stream.is_open()){
        cerr << "Error: failed to open " << test_img_file_name << endl;
        exit(1);
     }
    // Read magic number
    int magic_number;
    for(unsigned i = 0 ; i < 4; i++) {
        file_stream.read((char*)&magic_number,sizeof(int));
    }
    // Read test image
    data_type_t img;
    for(unsigned i = 0 ; i < test_set_size ; i++) {
        file_stream.read((char*)&img,sizeof(char));
        test_img_set[i] = img;
    }
}

// Read test label file
void mlp_t::read_test_label_file() {
    fstream file_stream;
    file_stream.open(test_label_file_name.c_str(), fstream::in|fstream::binary);

    if(!file_stream.is_open()){
        cerr<< "Error: failed to open" << test_label_file_name << endl;
        exit(1);
    }
    // Read magic number
    int magic_number;
    for(unsigned i = 0 ; i < 2; i++) {
        file_stream.read((char*)&magic_number,sizeof(int));
    }
    // Read test label
    data_type_t label;
    for(unsigned i = 0 ; i < test_set_size ; i++) {
        file_stream.read((char*)&label,sizeof(char));
        test_label_set[i] = label;
    }
}

// Read train image file
void mlp_t::read_train_img_file(){
    if(!require_training) return;

    fstream file_stream;
    file_stream.open(train_img_file_name.c_str(), fstream::in|fstream::binary);

    if(!file_stream.is_open()){
        cerr << "Error: failed to open" << train_img_file_name << endl;
        exit(1);
    }
    // Read magic number
    int magic_number;
    for(unsigned i = 0 ; i < 4; i++) {
        file_stream.read((char*)&magic_number,sizeof(int));
    }
    // Read train image
    data_type_t img;
    for(unsigned i = 0 ; i < train_set_size*num_neurons_in_input_layer ; i++) {
        file_stream.read((char*)&img,sizeof(char));
        train_img_set[i] = img;
    }
}

// Read train label file
void mlp_t::read_train_label_file(){
    if(!require_training) return;

    fstream file_stream;
    file_stream.open(train_label_file_name.c_str(), fstream::in|fstream::binary);

    if(!file_stream.is_open()){
        cerr<< "Error: failed to open" << train_label_file_name << endl;
        exit(1);
    }
    // Read magic number
    int magic_number;
    for(unsigned i = 0 ; i < 2; i++) {
        file_stream.read((char*)&magic_number,sizeof(int));
    }
    // Read train label
    data_type_t label;
    for(unsigned i = 0 ; i < train_set_size ; i++) {
        file_stream.read((char*)&label,sizeof(char));
        train_label_set[i] = label;
    }
}

// Initialize weights
void mlp_t::init_weights() {
    default_random_engine generator;
    normal_distribution <double> distribution(0.0, 0.01); // Mean = 0.0, Variance = 0.01
    for(unsigned i = 0; i < total_layers_index; i++){
        for(unsigned j = 0; j < num_neurons_per_layer[i+1] * (num_neurons_per_layer[i]+1); j++) {
			weights [i][j] = distribution(generator);
        }
    }
}

// Load weights from pre-trained weight file
void mlp_t::load_weights() {
    fstream file_stream;
    file_stream.open(weight_file_name.c_str(), fstream::in|fstream::binary);

    if(!file_stream.is_open()) {
        cerr << "Error: failed to open" << weight_file_name << endl;
        exit(1);
    }
    for(unsigned i =0; i < total_layers_index; i++) {
        for(unsigned j=0; j < num_neurons_per_layer[i+1] * (num_neurons_per_layer[i]+1); j++){
            file_stream >> weights [i][j];
        }
    }
}

// Convert big endian to little endian (for 32bit integer)
int mlp_t::big_to_little_endian_int32(int x) {
    int tmp = (((x << 8) & 0xFF00FF00) | (((x >> 8) & 0xFF00FF)));
    return ((tmp << 16) | (tmp >> 16));
}

double mlp_t::relu(const double x) {
    return max(x, double(0.0));
}

double mlp_t::drelu(const double x) {
    if(x > 0) return 1.0;
    else return 0.0;
}

void mlp_t::mlp_test() {
	int count = 0;
	for(unsigned i = 0; i < 10/*test_set_size*/; i++) {
		if(i % 1000 == 0) cout << i << "th is done." << endl;
		
		// Setting input image
		for(unsigned j = 0; j < num_neurons_in_input_layer; j++) {
			neuron[0][j] = test_img_set[i*num_neurons_in_input_layer+j];
		}
		// Setting Bias
		for(unsigned j = 0; j < total_layers_index; j++) {
			neuron[j][num_neurons_per_layer[j]] = 1.0;
		}
		
		inner_product(neuron, weights);
		softmax(neuron[total_layers_index]);
		cout << endl;
		for(unsigned j = 0; j < 10; j++) {
			cout << neuron[2][j] << " ";
		}
		cout << endl << train_label_set[i] << endl;
		cout << test_label_set[i] << endl;

		
		double max = 0.0;
		int max_index = 0;
		for(unsigned j = 0; j < num_neurons_per_layer[total_layers_index]; j++) {
			if(neuron[total_layers_index][j] > max) {
				max = neuron[total_layers_index][j];
				max_index = j;
			}
		}
		if(max_index == test_label_set[i]) {
			count++;
		}
	}
	cout << double(count) / double(test_set_size) << endl;
}


void mlp_t::mlp_training() {
	for(unsigned i = 0; i < 10/*train_set_size*/; i++) {
//#ifdef Debug
		if(i % 1000 == 0) {
			cout << i << "th is done." << endl;
			cout << loss << endl;
		}
		/*
		for(unsigned k = 0; k < 20; k++) {
			cout << weights[0][k] << " ";
			if(k!=0 && k%84==0) cout << endl;
		}
		cout << endl;
		*/

//#endifi
		// Initializing
		for(unsigned j = 0; j < total_layers_index; j++) {
			for(unsigned k = 0; k < num_neurons_per_layer[j]; k++) {
				neuron[j][k] = 0;
			}
		}

		// Setting input image
		for(unsigned j = 0; j < num_neurons_in_input_layer; j++) {
				neuron[0][j] = train_img_set[i*num_neurons_in_input_layer+j];
		}

		// Setting Bias
		for(unsigned j = 0; j < total_layers_index; j++) {
			neuron[j][num_neurons_per_layer[j]] = 1.0;
		}

		inner_product(neuron, weights);
		for(unsigned j = 0; j < 10; j++) {
			cout << neuron[2][j] << " ";
		}
		cout << endl;

		softmax(neuron[total_layers_index]);
		cout << endl;
		for(unsigned j = 0; j < 10; j++) {
			cout << neuron[2][j] << " ";
		}
		cout << endl << train_label_set[i] << endl;


		//Setting answer set
		for(unsigned k = 0; k < num_neurons_per_layer[total_layers_index]; k++) {
			if(k == unsigned(train_label_set[i])) {
				answer_set[k] = 1.0;
			}
			else {
				answer_set[k] = 0.0;
			}
		}

		loss = 0.0;
		for(unsigned k = 0; k < num_neurons_per_layer[total_layers_index]; k++) {
			loss -= answer_set[k] * log(neuron[total_layers_index][k]);
		}
		
		backward_propagation();
	}
}

/*
void mlp_t::forward_propagation() {
    for(unsigned i = 0; i < total_layers_index; i++) {
#ifdef DEBUG
        if(i % 1000 == 0) cout << i << "th is done." << endl;
#endif
        if(!require_training) {
			for(unsigned j = 0; j < num_neurons_in_input_layer; j++) {
				neurons_in_hidden_layer[0][j] = test_img_set[i*num_neurons_in_input_layer+j];
				inner_product(neurons_in_hidden_layer, weights, test_img_set);
            }
        }
        
        else {
			for(unsigned j = 0; j < num_neurons_in_input_layer; j++) {
				neurons_in_hidden_layer[0][j] = train_img_set[i*num_neurons_in_input_layer+j];
				inner_product(neurons_in_hidden_layer, weights, train_img_set);
            }
        }
        
        
        softmax(neurons_in_hidden_layer[total_layers_index]);
    }
}
*/

void mlp_t::inner_product(double **neurons, double **weight) {
    for(unsigned l = 0; l < total_layers_index; l++) {
        for(unsigned j = 0; j < num_neurons_per_layer[l+1]; j++) {
            double sum = 0.0;
            for(unsigned k = 0; k < num_neurons_per_layer[l]+1; k++){
                sum += weight[l][j*(num_neurons_per_layer[l]+1)+k] * neurons[l][k];
			}
            if(l+1 == total_layers_index) neurons[l+1][j] = sum;
            else neurons[l+1][j] = relu(sum);
        }
    }
}

void mlp_t::softmax(double *neurons) {
    double max = 0.0;
	for(unsigned l = 0; l < num_neurons_per_layer[total_layers_index]; l++) {
		if(max < neurons[l]) max = neurons[l];
	}

	double sum = 0.0001;
    for(unsigned l = 0; l < num_neurons_per_layer[total_layers_index]; l++) {
        sum += exp(neurons[l] - max);
    }

    for(unsigned l = 0; l < num_neurons_per_layer[total_layers_index]; l++) {
        neurons[l] = exp(neurons[l] - max) / sum;
    }
}

void mlp_t::backward_propagation() {
	// Initialize delta
	//cout << weights[1][500] << endl;
    for(unsigned k = 0; k < num_neurons_per_layer[total_layers_index]; k++) {
        delta[total_layers_index-1][k] = neuron[total_layers_index][k] * (1.0 - neuron[total_layers_index][k]) * (answer_set[k] - neuron[total_layers_index][k]);  
		//cout << delta[2][k] << " ";
	}
	//cout << endl;
	
    for(int l = num_layers-2; l >= 0; l--) {
        for(unsigned j = 0; j < num_neurons_per_layer[l]+1; j++) {
			double tmp = 0.0;
			for(unsigned k = 0; k < num_neurons_per_layer[l+1]; k++) {
				tmp += weights[l][k*num_neurons_per_layer[l]+j] * delta[l][k];
			}
			//delta[l][j] = drelu(neuron[l][j]) * tmp;
			if(l >= 1) delta[l-1][j] = drelu(neuron[l][j]) * tmp;

			// Update weights
			for(unsigned k = 0; k < num_neurons_per_layer[l+1]; k++) {
				weights[l][k*num_neurons_per_layer[l]+j] += learning_rate * neuron[l][j]* delta[l][k];
			}
		}
	}
	//cout << weights[1][500] << endl;
}

/*
            //if(j < num_neurons_per_layer[i+1] - 1) {
            // Calculate delta
            double tmp = 0.0;
            if(i + 1 == total_layers_index) {
                for(unsigned k = 0; k < num_neurons_per_layer[i+1]; k++) {
                    tmp += weights[i][j*num_neurons_per_layer[i]+k] * delta[i][k];
                }
            }
            else {
                for(unsigned k = 0; k < num_neurons_per_layer[i+1]-1; k++) {
                    tmp += weights[i][j*num_neurons_per_layer[i]+k] * delta[i][k];
                }
            }
            if(i != 0) delta[i][j] = drelu(neuron[i][j]) * tmp;
            else delta[i][j] = 0;
            }

                // Update weights
                if(i + 1 == num_layers - 1) {
                    for(unsigned k = 0; k < num_neurons_per_layer[i+1]; k++) {
                        weights[i][j*num_neurons_per_layer[i]+k] += learning_rate * neuron[i][j] * delta[i+1][k];
                    }
                }
                else {
                    for(unsigned k = 0; k < num_neurons_per_layer[i+1]-1; k++) {
                        weights[i][j*num_neurons_per_layer[i]+k] += learning_rate * neuron[i][j] * delta[i+1][k];
                    }
                }
        }
    }
}
*/

/*
void mlp_t::mlp_test() {
}

void mlp_t::mlp_train() {
}
*/
