/*****************************************************
   Multi-Level Perceptron (MLP) in C++
   Written by Intelligent Computing Systems Lab (ICSL)
   School of Electrical Engineering
   Yonsei University, Seoul,  South Korea
 *****************************************************/

#include <stdint.h>
#include <string>
#include <vector>

typedef uint8_t data_type_t;

// MLP layer types
enum MLP_LAYERS { NONE = 0, CONV, POOL, CLASS, NUM_LAYER_TYPES };

// MLP class
class mlp_t {
public:
    mlp_t();                                             // MLP constructor
    virtual ~mlp_t();                                    // MLP destructor
    
    void initialize(std::string m_config_file_name);     // Initialize MLP parameters
    void init_weights();
    void read_test_img_file();
    void read_test_label_file();
    void read_train_img_file();
    void read_train_label_file();
    void load_weights();
    //void forward_propagation();
    void backward_propagation();
    void mlp_test();
    void mlp_training();

    void inner_product(double **neuron, double **weights);
    void softmax(double *neuron); 
    
    int big_to_little_endian_int32(int x);
    
    double relu(double x);
    double drelu(double x);
private:
    double **neuron;
    unsigned width, length;
    bool require_training;

    std::string config_file_name;                        // Configuration file name
    std::string test_img_file_name;                      // Test img file for inferencing
    std::string test_label_file_name;                    // Test label file for inferencing
    std::string train_img_file_name;                     // MLP train img file name
    std::string train_label_file_name;                   // MLP train label file name 
    std::string weight_file_name;                        // Pre-trained weight file name 

    unsigned num_layers;
    unsigned total_layers_index;
	unsigned num_neurons_in_input_layer;
    unsigned num_neurons_in_hidden_layer;
    unsigned num_neurons_in_output_layer;
    unsigned *num_neurons_per_layer;
    unsigned test_set_size;
    unsigned train_set_size;
    double *test_img_set;
    double *train_img_set;
    double *test_label_set;
    double *train_label_set;
	double *answer_set;
    double **weights;
    double **delta;
    double learning_rate;
	double loss;
};
