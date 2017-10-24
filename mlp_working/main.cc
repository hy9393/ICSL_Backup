/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include "mlp.h"

using namespace std;

void print_usage(char *exec) {
    cout << "Usage: " << exec                                   << endl
         << "       -config <required: mlp config file>"        << endl;
//         << "       -test_img <required: mlp test file>"        << endl
//         << "       -test_label <required: mlp test file>"      << endl
//         << "       -train_img <optional: mlp training file>"   << endl
//         << "       -train_label <optional: mlp training file>" << endl
//         << "       -weight <optional: mlp weight file>"        << endl;
    exit(1);
}

int main(int argc, char **argv) {
    // Check # of input arguments.
    if(argc < 3) { print_usage(argv[0]); }

    // Parse input arguments.
    string config_file_name, weight_file_name;
    string test_img_file_name, test_label_file_name;
    string train_img_file_name, train_label_file_name;
    
    for(int i = 1; i < argc; i++) {
        if(!strcasecmp(argv[i],"-config")) {
            config_file_name = argv[++i];
        }
        /*
        else if(!strcasecmp(argv[i],"-test_img")) {
            test_img_file_name = argv[++i];
        }
        else if(!strcasecmp(argv[i],"-test_label")) {
            test_label_file_name = argv[++i];
        }
        else if(!strcasecmp(argv[i],"-train_img")) {
            train_img_file_name = argv[++i];
        }
        else if(!strcasecmp(argv[i],"-train_label")){
            train_label_file_name = argv[++i];
        }
        else if(!strcasecmp(argv[i],"-weight")) {
            weight_file_name = argv[++i];
        }*/
        else {
            cout << "Error: unknown option " << argv[i] << endl;
            exit(1);
        }
    }

    if(!config_file_name.size()) { print_usage(argv[0]); }

    /*
    if(!config_file_name.size() || !test_img_file_name.size() || !test_label_file_name.size() ||
      (!weight_file_name.size() && (!train_img_file_name.size() || !train_label_file_name.size()))) {
        print_usage(argv[0]);
    }*/



    mlp_t *mlp = new mlp_t(); //NULL, 0, 0, NULL, NULL, NULL, NULL ); 
    
    mlp->initialize(config_file_name); //, test_img_file_name, test_label_file_name, train_img_file_name, train_label_file_name, weight_file_name);
	mlp->read_test_img_file();
    mlp->read_test_label_file();
    mlp->read_train_img_file();
    mlp->read_train_label_file();

	//mlp->mlp_training();
	mlp->mlp_test();

    #ifdef DEBUG
    for(unsigned i = 0; i < num_layers-1; i++) {
            for(unsigned j = 0; j < num_neurons_per_layer[i] * num_neurons_per_layer[i+1]; j++) {
                cout << "weight["<<i<<","<<j<<"] = "<<weights[i][j]<<endl;
            }
    }
    #endif
    delete mlp;

    return 0;
}
