#include <stdio.h>
#include "gpu_nn.h"

/*
Allocate the memory required by the GPU solver.
Does not handle memory allocation for the data we load in.
*/

void Send_To_Device(float **h_neuron_bias,  float **d_neuron_bias,
                     short **h_neuron_forward_recieve_id, short **d_neuron_forward_recieve_id,
                     short **h_neuron_forward_receive_start, short **d_neuron_forward_receive_start,
                     float ** h_neuron_forward_recieve_weight, float ** d_neuron_forward_recieve_weight,
                     short **h_layer_start, short **d_layer_start,
                     short **h_layer_neuron_type, short **d_layer_neuron_type,
                     float *h_training_data, float **d_training_data, size_t training_data_size,
                     float *h_training_classification, float **d_training_classification, size_t training_class_size,
                     int NO_NEURONS, int NO_WEIGHTS, int NO_LAYERS) {

    // Grab a error type
    cudaError_t Error;

    // Send Neuron data to GPU
    size_t size = NO_NEURONS*sizeof(float);
    Error = cudaMemcpy(*d_neuron_bias, *h_neuron_bias, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_neuron_bias -> d_neuron_bias) = %s\n", cudaGetErrorString(Error));
    
    // Send weight information to GPU
    size = NO_WEIGHTS*sizeof(float);
    Error = cudaMemcpy(*d_neuron_forward_recieve_weight, *h_neuron_forward_recieve_weight, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_neuron_forward_recieve_weight -> d_neuron_forward_recieve_weight) = %s\n", cudaGetErrorString(Error));    
    
    size = NO_WEIGHTS*sizeof(short);
    Error = cudaMemcpy(*d_neuron_forward_recieve_id, *h_neuron_forward_recieve_id, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_neuron_forward_recieve_id -> d_neuron_forward_recieve_id) = %s\n", cudaGetErrorString(Error));    

    size = (NO_NEURONS+1)*sizeof(short);
    Error = cudaMemcpy(*d_neuron_forward_receive_start, *h_neuron_forward_receive_start, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_neuron_forward_receive_start -> d_neuron_forward_receive_start) = %s\n", cudaGetErrorString(Error));

    size = (NO_LAYERS+1)*sizeof(short);
    Error = cudaMemcpy(*d_layer_start, *h_layer_start, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_layer_start -> d_layer_start) = %s\n", cudaGetErrorString(Error));

    size = NO_LAYERS*sizeof(short);
    Error = cudaMemcpy(*d_layer_neuron_type, *h_layer_neuron_type, size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_layer_neuron_type -> d_layer_neuron_type) = %s\n", cudaGetErrorString(Error));   

    // Training Data
    Error = cudaMemcpy(*d_training_data, h_training_data, training_data_size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_training_data -> d_training_data) = %s\n", cudaGetErrorString(Error)); 

    Error = cudaMemcpy(*d_training_classification, h_training_classification, training_class_size, cudaMemcpyHostToDevice); 
    printf("CUDA error (memcpy h_training_classification -> d_training_classification) = %s\n", cudaGetErrorString(Error)); 

}




void Allocate_Memory(float **h_neuron_bias,  float **d_neuron_bias,
                     float **h_neuron_input, float **d_neuron_input,
                     float **h_neuron_output, float **d_neuron_output,
                     float **h_neuron_delta, float **d_neuron_delta,
                     short **h_neuron_forward_recieve_id, short **d_neuron_forward_recieve_id,
                     short **h_neuron_forward_receive_start, short **d_neuron_forward_receive_start,
                     float ** h_neuron_forward_recieve_weight, float ** d_neuron_forward_recieve_weight,
                     short **h_layer_start, short **d_layer_start,
                     short **h_layer_neuron_type, short **d_layer_neuron_type,
                     float **d_training_data, size_t training_data_size,
                     float **d_training_classification, size_t training_class_size,
                     int NO_NEURONS, int NO_WEIGHTS, int NO_LAYERS) {

    size_t size = NO_NEURONS*sizeof(float);
    *h_neuron_bias = (float*)malloc(size);
    *h_neuron_input = (float*)malloc(size);
    *h_neuron_output = (float*)malloc(size);
    *h_neuron_delta = (float*)malloc(size);

    cudaError_t Error;
    Error = cudaMalloc((void**)d_neuron_bias, size); 
    printf("CUDA error (malloc d_neuron_bias) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_neuron_input, size); 
    printf("CUDA error (malloc d_neuron_input) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_neuron_output, size); 
    printf("CUDA error (malloc d_neuron_output) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_neuron_delta, size); 
    printf("CUDA error (malloc d_neuron_delta) = %s\n", cudaGetErrorString(Error));

    size = NO_WEIGHTS*sizeof(short);
    *h_neuron_forward_recieve_id = (short*)malloc(size);
    Error = cudaMalloc((void**)d_neuron_forward_recieve_id, size); 
    printf("CUDA error (malloc d_neuron_forward_recieve_id) = %s\n", cudaGetErrorString(Error));

    size = NO_WEIGHTS*sizeof(float);
    *h_neuron_forward_recieve_weight = (float*)malloc(size);
    Error = cudaMalloc((void**)d_neuron_forward_recieve_weight, size); 
    printf("CUDA error (malloc h_neuron_forward_recieve_weight) = %s\n", cudaGetErrorString(Error));

    size = (NO_NEURONS+1)*sizeof(short);
    *h_neuron_forward_receive_start = (short*)malloc(size);
    Error = cudaMalloc((void**)d_neuron_forward_receive_start, size); 
    printf("CUDA error (malloc d_neuron_forward_receive_start) = %s\n", cudaGetErrorString(Error));
    
    size = NO_LAYERS*sizeof(short);
    *h_layer_neuron_type = (short*)malloc(size);
    Error = cudaMalloc((void**)d_layer_neuron_type, size); 
    printf("CUDA error (malloc d_layer_neuron_type) = %s\n", cudaGetErrorString(Error));

    size = (NO_LAYERS+1)*sizeof(short);
    *h_layer_start = (short*)malloc(size);
    Error = cudaMalloc((void**)d_layer_start, size); 
    printf("CUDA error (malloc d_layer_start) = %s\n", cudaGetErrorString(Error));

    // Detect the size of the training data sets, and allocate that much in Cuda
    Error = cudaMalloc((void**)d_training_data, training_data_size); 
    printf("CUDA error (malloc d_training_data) = %s\n", cudaGetErrorString(Error));

    Error = cudaMalloc((void**)d_training_classification, training_data_size); 
    printf("CUDA error (malloc d_training_classification) = %s\n", cudaGetErrorString(Error));

}

void Free_Memory(float **h_neuron_bias,  float **d_neuron_bias,
                     float **h_neuron_input, float **d_neuron_input,
                     float **h_neuron_output, float **d_neuron_output,
                     float **h_neuron_delta, float **d_neuron_delta,
                     short **h_neuron_forward_recieve_id, short **d_neuron_forward_recieve_id,
                     short **h_neuron_forward_receive_start, short **d_neuron_forward_receive_start,
                     float **h_neuron_forward_recieve_weight, float **d_neuron_forward_recieve_weight,
                     short **h_layer_start, short **d_layer_start,
                     short **h_layer_neuron_type, short **d_layer_neuron_type,
                     float **d_training_data, float **d_training_classification) {

    if (*h_neuron_bias) free(*h_neuron_bias);
    if (*h_neuron_input) free(*h_neuron_input);
    if (*h_neuron_output) free(*h_neuron_output);
    if (*h_neuron_delta) free(*h_neuron_delta);
    if (*h_neuron_forward_recieve_id) free(*h_neuron_forward_recieve_id);
    if (*h_neuron_forward_receive_start) free(*h_neuron_forward_receive_start);
    if (*h_neuron_forward_recieve_weight) free(*h_neuron_forward_recieve_weight);
    if (*h_layer_start) free(*h_layer_start);
    if (*h_layer_neuron_type) free(*h_layer_neuron_type);   

    if (*d_neuron_bias) cudaFree(*d_neuron_bias);
    if (*d_neuron_input) cudaFree(*d_neuron_input);
    if (*d_neuron_output) cudaFree(*d_neuron_output);
    if (*d_neuron_delta) cudaFree(*d_neuron_delta);
    if (*d_neuron_forward_recieve_id) cudaFree(*d_neuron_forward_recieve_id);
    if (*d_neuron_forward_receive_start) cudaFree(*d_neuron_forward_receive_start);
    if (*d_neuron_forward_recieve_weight) cudaFree(*d_neuron_forward_recieve_weight);

    if (*d_layer_start) cudaFree(*d_layer_start);
    if (*d_layer_neuron_type) cudaFree(*d_layer_neuron_type);
    if (*d_training_data) cudaFree(*d_training_data);
    if (*d_training_classification) cudaFree(*d_training_classification);    
}


void Prepare_Network_Size(short *network_layout, short *no_layers, short *no_weights, short *no_neurons) {
    *no_layers = sizeof(network_layout) / sizeof(network_layout[0])-1;
    *no_weights = 0;
    *no_neurons = 0;
    printf("No. of layers = %d\n", *no_layers);
    for (short layer = 0; layer < *no_layers; layer++) {
        printf("Found %d neurons in layer %d\n", network_layout[layer], layer);
        *no_neurons = *no_neurons + network_layout[layer];
    }
    // Compute the number of weights
    for (short layer = 1; layer < *no_layers; layer++) {
        *no_weights += network_layout[layer]*network_layout[layer-1];
    }

    printf("Found a total number of %d neurons\n", *no_neurons);
    printf("Found a total number of %d weights\n", *no_weights);
}

void Prepare_Network_Structure(short *layer_start, short *layer_neuron_type, short *network_layout, 
                               short *neuron_forward_receive_start, short *neuron_forward_recieve_id,
                               float *neuron_bias, float *neuron_forward_recieve_weight,
                               short no_layers, short no_neurons) {

    // The first layer is an input layer (type 0)    
    layer_neuron_type[0] = 0;
    // The last layer is an output layer (type 2)
    layer_neuron_type[no_layers-1] = 2;

    // The remaining types are inner (hidden) layers.
    for (short layer = 1; layer < (no_layers-1); layer++) {
        layer_neuron_type[layer] = 1;
    }

    layer_start[0] = 0;
    // Now to properly compose the layer starts
    for (short layer = 0; layer < no_layers; layer++) {
        layer_start[layer+1] = layer_start[layer] + network_layout[layer];
    }

    // Build up the network now
    // short layer_start[NO_LAYERS+1] = {0, 4, 9, 12};
    // short layer_neuron_type[NO_LAYERS] = {0, 1, 2}; // 0 = input, 1 = hidden, 2 = output
    // short neuron_forward_recieve_id[] =    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8};
    // short neuron_forward_receive_start[] = {0, 0, 0, 0, 0, 4, 8, 12, 16, 20, 25, 30, 35};
    neuron_forward_receive_start[0] = 0;
    short cumulative_sum_neurons = 0;
    short cumulative_input_sum = 0;
    short weight_count = 0;
    for (short layer = 0; layer <= no_layers; layer++) {
        // And we iterate over the neurons in this layer
        for (short layer_neuron = 0; layer_neuron < (layer_start[layer+1] - layer_start[layer]); layer_neuron++) {
            short neuron_id = layer_start[layer] + layer_neuron;
            if (layer_neuron_type[layer] == 0) {
                // This is an input layer
                neuron_forward_receive_start[neuron_id+1] = 0; // There are no inputs
            } else {
                short no_inputs = network_layout[layer-1];
                for (short input = 0; input < no_inputs; input++) {
                    short source_id = cumulative_sum_neurons + input;
                    neuron_forward_recieve_id[weight_count] = source_id;
                    weight_count++;
                }
                cumulative_input_sum += no_inputs;
                neuron_forward_receive_start[neuron_id+1] = cumulative_input_sum;
            }
        }
        if (layer_neuron_type[layer] > 0) {
            cumulative_sum_neurons += layer_start[layer]; // Keeps track of the id of the last neuron in the previous layer
        }
    }

    // Initialise the bias and weights values
    srand(time(NULL));
    for (int i = 0; i < no_neurons; i++) {
        neuron_bias[i] = (float)rand()/RAND_MAX;
    }
    for (int i = 0; i < weight_count; i++) {
        neuron_forward_recieve_weight[i] = (float)rand()/RAND_MAX;
    }

}  





void Train_Network(float *h_training_data, size_t training_data_size, float *h_training_classification, size_t training_class_size, short *network, int NO_SAMPLES, int NO_EPOCHS, float learning_rate) {

    float *h_neuron_bias, *d_neuron_bias;
    float *h_neuron_input, *d_neuron_input;
    float *h_neuron_output, *d_neuron_output;
    float *h_neuron_delta, *d_neuron_delta;
    short *h_neuron_forward_recieve_id, *d_neuron_forward_recieve_id;
    short *h_neuron_forward_receive_start, *d_neuron_forward_receive_start;
    float *h_neuron_forward_recieve_weight, *d_neuron_forward_recieve_weight;
    short *h_layer_neuron_type, *d_layer_neuron_type;
    short *h_layer_start, *d_layer_start;
    short no_weights, no_layers, no_neurons;

    // Prepare memory for the training data and classification
    float *d_training_data, *d_training_classification;

    Prepare_Network_Size(network, &no_layers, &no_weights, &no_neurons);

    printf("Idenfified %d layers, %d neurons and %d weights\n", no_layers, no_neurons, no_weights);

    Allocate_Memory(&h_neuron_bias,  &d_neuron_bias,
                     &h_neuron_input, &d_neuron_input,
                     &h_neuron_output, &d_neuron_output,
                     &h_neuron_delta, &d_neuron_delta,
                     &h_neuron_forward_recieve_id, &d_neuron_forward_recieve_id,
                     &h_neuron_forward_receive_start, &d_neuron_forward_receive_start,
                     &h_neuron_forward_recieve_weight, &d_neuron_forward_recieve_weight,
                     &h_layer_start, &d_layer_start,
                     &h_layer_neuron_type, &d_layer_neuron_type,
                     &d_training_data, training_data_size,
                     &d_training_classification, training_class_size,
                     no_neurons, no_weights, no_layers);

    Prepare_Network_Structure(h_layer_start, h_layer_neuron_type, network,
                             h_neuron_forward_receive_start, h_neuron_forward_recieve_id, 
                             h_neuron_bias, h_neuron_forward_recieve_weight,
                             no_layers, no_neurons);
    printf("Training network\n");


    // Send things to the GPU
    Send_To_Device(&h_neuron_bias,  &d_neuron_bias,
                   &h_neuron_forward_recieve_id, &d_neuron_forward_recieve_id,
                   &h_neuron_forward_receive_start, &d_neuron_forward_receive_start,
                   &h_neuron_forward_recieve_weight, &d_neuron_forward_recieve_weight,
                   &h_layer_start, &d_layer_start,
                   &h_layer_neuron_type, &d_layer_neuron_type,
                   h_training_data, &d_training_data, training_data_size,
                   h_training_classification, &d_training_classification, training_class_size,
                   no_neurons, no_weights, no_layers);

    // 

    for (int epoch = 0; epoch < NO_EPOCHS; epoch++) {
        // Employ Data Parallelism
        // Each thread will train (using forward and backward propagation) one sample

        int threads_per_block = 32;
        int no_blocks = NO_SAMPLES/threads_per_block;


    }




    Free_Memory(&h_neuron_bias,  &d_neuron_bias,
        &h_neuron_input, &d_neuron_input,
        &h_neuron_output, &d_neuron_output,
        &h_neuron_delta, &d_neuron_delta,
        &h_neuron_forward_recieve_id, &d_neuron_forward_recieve_id,
        &h_neuron_forward_receive_start, &d_neuron_forward_receive_start,
        &h_neuron_forward_recieve_weight, &d_neuron_forward_recieve_weight,
        &h_layer_start, &d_layer_start,
        &h_layer_neuron_type, &d_layer_neuron_type,
        &d_training_data, &d_training_classification);

}