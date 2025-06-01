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
                     int NO_NEURONS, int NO_WEIGHTS, int NO_LAYERS, int NO_SAMPLES);

void Free_Memory(float **h_neuron_bias,  float **d_neuron_bias,
                     float **h_neuron_input, float **d_neuron_input,
                     float **h_neuron_output, float **d_neuron_output,
                     float **h_neuron_delta, float **d_neuron_delta,
                     short **h_neuron_forward_recieve_id, short **d_neuron_forward_recieve_id,
                     short **h_neuron_forward_receive_start, short **d_neuron_forward_receive_start,
                     float **h_neuron_forward_recieve_weight, float **d_neuron_forward_recieve_weight,
                     short **h_layer_start, short **d_layer_start,
                     short **h_layer_neuron_type, short **d_layer_neuron_type,
                     float **d_training_data, float **d_training_classification);

void Prepare_Network_Size(short *network_layout, short *no_layers, short *no_weights, short *no_neurons);

void Prepare_Network_Structure(short *layer_start, short *layer_neuron_type, short *network_layout, 
                               short *neuron_forward_receive_start, short *neuron_forward_recieve_id,
                               float *neuron_bias, float *neuron_forward_recieve_weight,
                               short no_layers, short no_neurons);

void Train_Network(float *h_training_data, size_t training_data_size, float *h_training_classification, size_t training_class_size, short *network, int NO_SAMPLES, int NO_EPOCHS, float learning_rate);