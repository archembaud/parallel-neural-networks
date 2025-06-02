#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void Prepare_Network_Size(short *network_layout, short *no_layers, short *no_weights, short *no_neurons) {

    *no_weights = 0;
    *no_neurons = 0;

    short layer_neurons = 1;
    short layer_count = 0; 
    while (layer_neurons > 0) {
        layer_neurons = network_layout[layer_count];
        layer_count++;
    }
    *no_layers = layer_count-1;  
    for (short layer = 0; layer < *no_layers; layer++) {
        *no_neurons = *no_neurons + network_layout[layer];
    }
    // Compute the number of weights
    for (short layer = 1; layer < *no_layers; layer++) {
        *no_weights += network_layout[layer]*network_layout[layer-1];
    }
}


void Prepare_Network_Structure(short *layer_start, short *layer_neuron_type, short *network_layout, 
                               short *neuron_forward_receive_start, short *neuron_forward_recieve_id,
                               float *neuron_bias, float *neuron_forward_recieve_weight,
                               short no_layers, short no_neurons, short no_weights) {

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
            cumulative_sum_neurons = layer_start[layer]; // Keeps track of the id of the last neuron in the previous layer
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



void Allocate_Memory(float **h_neuron_bias,
                     float **h_neuron_input,
                     float **h_neuron_output,
                     float **h_neuron_delta,
                     short **h_neuron_forward_recieve_id,
                     short **h_neuron_forward_receive_start,
                     float ** h_neuron_forward_recieve_weight,
                     short **h_layer_start,
                     short **h_layer_neuron_type,
                     int NO_NEURONS, int NO_WEIGHTS, int NO_LAYERS, int NO_SAMPLES) {


    size_t size = NO_NEURONS*sizeof(float);
    *h_neuron_bias = (float*)malloc(size);

    size = NO_NEURONS*NO_SAMPLES*sizeof(float);
    *h_neuron_input = (float*)malloc(size);
    *h_neuron_output = (float*)malloc(size);
    *h_neuron_delta = (float*)malloc(size);

    size = NO_WEIGHTS*sizeof(short);
    *h_neuron_forward_recieve_id = (short*)malloc(size);

    size = NO_WEIGHTS*sizeof(float);
    *h_neuron_forward_recieve_weight = (float*)malloc(size);

    size = (NO_NEURONS+1)*sizeof(short);
    *h_neuron_forward_receive_start = (short*)malloc(size);
    
    size = NO_LAYERS*sizeof(short);
    *h_layer_neuron_type = (short*)malloc(size);

    size = (NO_LAYERS+1)*sizeof(short);
    *h_layer_start = (short*)malloc(size);

}

void Free_Memory(float **h_neuron_bias,
                     float **h_neuron_input,
                     float **h_neuron_output,
                     float **h_neuron_delta,
                     short **h_neuron_forward_recieve_id,
                     short **h_neuron_forward_receive_start,
                     float **h_neuron_forward_recieve_weight,
                     short **h_layer_start,
                     short **h_layer_neuron_type) {

    if (*h_neuron_bias) free(*h_neuron_bias);
    if (*h_neuron_input) free(*h_neuron_input);
    if (*h_neuron_output) free(*h_neuron_output);
    if (*h_neuron_delta) free(*h_neuron_delta);
    if (*h_neuron_forward_recieve_id) free(*h_neuron_forward_recieve_id);
    if (*h_neuron_forward_receive_start) free(*h_neuron_forward_receive_start);
    if (*h_neuron_forward_recieve_weight) free(*h_neuron_forward_recieve_weight);
    if (*h_layer_start) free(*h_layer_start);
    if (*h_layer_neuron_type) free(*h_layer_neuron_type);   

}

float Activation_Function(float input) {
    // Use the sigmoid function
    return (1.0/(1.0 + expf(-input)));
}


void Run_Network(float *input, float *computed_output,
                 float *neuron_delta, float *neuron_input, float *neuron_output, float *neuron_bias,
                 short *layer_start, short *layer_neuron_type,
                 short *neuron_forward_recieve_id, short *neuron_forward_receive_start, float *neuron_forward_recieve_weight,
                 short NO_LAYERS, short NO_NEURONS) {

    // Reset our deltas
    for (int i = 0; i < NO_NEURONS; i++) {
        neuron_delta[i] = 0.0;
        neuron_input[i] = 0.0;
        neuron_output[i] = 0.0;
    }

    // Moving forward
    for (int layer = 0; layer < NO_LAYERS; layer++) {
        // Now, process the neurons in each layer
        for (int layer_neuron = 0; layer_neuron < (layer_start[layer+1] - layer_start[layer]); layer_neuron++) {
            int neuron_id = layer_start[layer] + layer_neuron;
            // For each neuron, we need to iterate over its sources
            // If there are no sources, check to see if this is an input
            int no_sources = neuron_forward_receive_start[neuron_id+1] - neuron_forward_receive_start[neuron_id];
            if (no_sources == 0) {
                // Probably an input. Double check anyway.
                if (layer_neuron_type[layer] == 0) {
                    // This is an input.
                    // Handle the input.
                    neuron_input[neuron_id] = input[neuron_id];
                    // Also, we won't put a bias on neurons in the input layer
                    neuron_output[neuron_id] = neuron_input[neuron_id];
                } else {
                    printf("ERROR: neuron %d has no inputs\n", neuron_id);
                }
            } else {
                // We have a middle layer, or an output layer.
                float neuron_input_sum = 0.0;
                for (int source = 0; source < no_sources; source++) {
                    int source_id = neuron_forward_recieve_id[neuron_forward_receive_start[neuron_id] + source];
                    // Weights are stored in the same structure as the source node
                    neuron_input_sum += neuron_output[source_id]*neuron_forward_recieve_weight[neuron_forward_receive_start[neuron_id] + source];
                }
                // Set the input (for record keeping) then set the output
                neuron_input[neuron_id] = neuron_input_sum;
                // Add a bias
                neuron_input[neuron_id] += neuron_bias[neuron_id];
                // Apply an activation function and compute an output
                neuron_output[neuron_id] = Activation_Function(neuron_input_sum);
            }            
        }
    }

    // Now to return the output vector; these are the last 3 neurons
    computed_output[0] = neuron_output[NO_NEURONS-3];
    computed_output[1] = neuron_output[NO_NEURONS-2];
    computed_output[2] = neuron_output[NO_NEURONS-1];
}

void Train_Network_using_single_sample(float *input, float *output, float *computed_output, float learning_rate,
                                    float *neuron_delta, float *neuron_input, float *neuron_output, float *neuron_bias,
                                    short *layer_start, short *layer_neuron_type,
                                    short *neuron_forward_recieve_id, short *neuron_forward_receive_start, float *neuron_forward_recieve_weight,
                                    short NO_LAYERS, short NO_NEURONS) {

    // Run the network forward
    Run_Network(input, computed_output,
                neuron_delta, neuron_input, neuron_output, neuron_bias,
                layer_start, layer_neuron_type,
                neuron_forward_recieve_id, neuron_forward_receive_start, neuron_forward_recieve_weight,
                NO_LAYERS, NO_NEURONS);

    // Now we need to run it backwards (backwards propagation)
    // Start by computing the deltas at the output layer.
    // So our outer loop is run in reverse
    for (int layer = (NO_LAYERS-1); layer >= 0; layer--) {
        for (int layer_neuron = 0; layer_neuron < (layer_start[layer+1] - layer_start[layer]); layer_neuron++) {
            int neuron_id = layer_start[layer] + layer_neuron;
            if (layer_neuron_type[layer] == 2) {
                // This is the output layer
                // There are many possible different methods for computing this delta
                // Sigmoid Activation approach
                neuron_delta[neuron_id] = (output[layer_neuron] - computed_output[layer_neuron])
                *computed_output[layer_neuron]*(1.0 - computed_output[layer_neuron]);

                // Now while we are here, propagate the deltas through to the source neurons
                int no_sources = neuron_forward_receive_start[neuron_id+1] - neuron_forward_receive_start[neuron_id];
                for (int source = 0; source < no_sources; source++) {
                    int source_id = neuron_forward_recieve_id[neuron_forward_receive_start[neuron_id] + source];
                    // Update the delta of that source - can parallelize this loop
                    float sigma_dash = neuron_output[neuron_id]*(1.0 - neuron_output[neuron_id]);
                    int weight_index = neuron_forward_receive_start[neuron_id] + source;
                    neuron_delta[source_id] += neuron_delta[neuron_id]*neuron_forward_recieve_weight[neuron_forward_receive_start[neuron_id] + source]*sigma_dash;
                }
                // Check the value of these before continuing
                for (int source = 0; source < no_sources; source++) {
                    int source_id = neuron_forward_recieve_id[neuron_forward_receive_start[neuron_id] + source];
                }
            } else {
                // Hidden and input layers
                int neuron_id = layer_start[layer] + layer_neuron;
                // We need to propagate the delta back to sources
                int no_sources = neuron_forward_receive_start[neuron_id+1] - neuron_forward_receive_start[neuron_id];
                for (int source = 0; source < no_sources; source++) {
                    int source_id = neuron_forward_recieve_id[neuron_forward_receive_start[neuron_id] + source];
                    int weight_index = neuron_forward_receive_start[neuron_id] + source;
                    float sigma_dash = neuron_output[neuron_id]*(1.0 - neuron_output[neuron_id]);
                    float d_delta = neuron_delta[neuron_id]*neuron_forward_recieve_weight[neuron_forward_receive_start[neuron_id] + source]*sigma_dash;
                    neuron_delta[source_id] += d_delta;
                }
            }
        }
    }

    // Now we can run forward and update weights and bias values
    for (int layer = 0; layer < NO_LAYERS; layer++) {
        for (int layer_neuron = 0; layer_neuron < (layer_start[layer+1] - layer_start[layer]); layer_neuron++) {
            int neuron_id = layer_start[layer] + layer_neuron;
            int no_sources = neuron_forward_receive_start[neuron_id+1] - neuron_forward_receive_start[neuron_id];
            if (no_sources == 0) {
                // This is an input layer.
                neuron_bias[neuron_id] = 0.0; // Not really required
            } else {
                float sigma_dash = neuron_output[neuron_id]*(1.0 - neuron_output[neuron_id]);
                float change_in_bias = learning_rate * neuron_delta[neuron_id] * sigma_dash;
                neuron_bias[neuron_id] += change_in_bias;
                // We are in an output layer, which means we are updating the weights between here and the middle layer
                for (int source = 0; source < no_sources; source++) {
                    int source_id = neuron_forward_recieve_id[neuron_forward_receive_start[neuron_id] + source];
                    int weight_index = neuron_forward_receive_start[neuron_id] + source;
                    float weight_change = learning_rate*neuron_delta[neuron_id]*sigma_dash*neuron_output[source_id];
                    neuron_forward_recieve_weight[weight_index] += weight_change;
                }
            }
        }
    }
}

int Check_Prediction(float *computed_output, float *expected_output) {
    // Check each of the outputs
    // This is pretty custom to the data being loaded
    int max_category_computed = -1;
    float computed_max = -1.0;
    // Iterate over the outputs
    for (int i = 0; i < 3; i++) {
        if (computed_output[i] > computed_max) {
            computed_max = computed_output[i];
            max_category_computed = i;
        }
    }
    // Check
    if (expected_output[max_category_computed] == 1.0) {
        return 1;
    } else {
        return 0;
    }
}

void Train_Network(float *h_training_data, float *h_training_classification, short *network, int NO_SAMPLES, int NO_EPOCHS, float learning_rate) {

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
    short no_inputs = network[0];
    // Open files to save the results (and history of training)
    FILE *pBiasFile;
    pBiasFile = fopen("Bias.txt", "w");
    FILE *pWeightFile;
    pWeightFile = fopen("Weights.txt", "w");
    FILE *pAccuracyFile;
    pAccuracyFile = fopen("Accuracy.txt", "w");

    // Prepare memory for the training data and classification
    float *d_training_data, *d_training_classification;

    Prepare_Network_Size(network, &no_layers, &no_weights, &no_neurons);
    short no_outputs = network[no_layers-1];

    // Allocate memory
    Allocate_Memory(&h_neuron_bias,
                     &h_neuron_input,
                     &h_neuron_output,
                     &h_neuron_delta,
                     &h_neuron_forward_recieve_id,
                     &h_neuron_forward_receive_start,
                     &h_neuron_forward_recieve_weight,
                     &h_layer_start,
                     &h_layer_neuron_type,
                     no_neurons, no_weights, no_layers, NO_SAMPLES);

    Prepare_Network_Structure(h_layer_start, h_layer_neuron_type, network,
                             h_neuron_forward_receive_start, h_neuron_forward_recieve_id, 
                             h_neuron_bias, h_neuron_forward_recieve_weight,
                             no_layers, no_neurons, no_weights);


    // Train the network
    for (int epoch = 0; epoch < NO_EPOCHS; epoch++) {
        // Train our AI using all of the samples
        int no_successful_predictions = 0;
        for (int sample = 0; sample < NO_SAMPLES; sample++) {
            // This data has 4 inputs - 4 measurements of the plant - we can use for classification
            // Collect the input data for each sample
            float sample_input[4] = {h_training_data[sample*4+0], h_training_data[sample*4+1], h_training_data[sample*4+2], h_training_data[sample*4+3]};
            // Collect the classification data for each sample 
            float sample_classification[3] = {h_training_classification[sample*3+0], h_training_classification[sample*3+1], h_training_classification[sample*3+2]};
            float computed_results[3];
            // Train the network using this sample
            Train_Network_using_single_sample(sample_input, sample_classification, computed_results, 0.1,
                                              h_neuron_delta, h_neuron_input, h_neuron_output, h_neuron_bias,
                                              h_layer_start, h_layer_neuron_type,
                                              h_neuron_forward_recieve_id, h_neuron_forward_receive_start, h_neuron_forward_recieve_weight,
                                              no_layers, no_neurons);

            no_successful_predictions += Check_Prediction(computed_results, sample_classification);
        }
        // Show training progress
        printf("Epoch %d of %d - accuracy = %g\n", epoch, NO_EPOCHS, (float)no_successful_predictions/NO_SAMPLES);
        fprintf(pAccuracyFile, "%d\t%g\n", epoch, (float)no_successful_predictions/NO_SAMPLES);
        // Now I want to record the weights and the bias values
        for (int i = 0; i < no_neurons; i++) {
            fprintf(pBiasFile, "%d\t%d\t%g\n",epoch, i, h_neuron_bias[i]);
        }
        for (int i = 0; i < no_weights; i++) {
            fprintf(pWeightFile, "%d\t%d\t%g\n",epoch, i, h_neuron_forward_recieve_weight[i]);
        }
    }


    /*
        Randomly pick one of these for display    
    */
    int sample = rand() % 150;
    float sample_input[4] = {h_training_data[sample*4+0], h_training_data[sample*4+1], h_training_data[sample*4+2], h_training_data[sample*4+3]};
    // Collect the classification data for each sample 
    float sample_classification[3] = {h_training_classification[sample*3+0], h_training_classification[sample*3+1], h_training_classification[sample*3+2]};
    float computed_results[3];
    // Run the network forward
    Run_Network(sample_input, computed_results,
                h_neuron_delta, h_neuron_input, h_neuron_output, h_neuron_bias,
                h_layer_start, h_layer_neuron_type,
                h_neuron_forward_recieve_id, h_neuron_forward_receive_start, h_neuron_forward_recieve_weight,
                no_layers, no_neurons);
    // Show the results
    printf("=======================================\n");
    printf("Checking randomly selected sample %d\n", sample);
    printf("Expected outputs = %g, %g, %g\n", sample_classification[0], sample_classification[1], sample_classification[2]);
    printf("Measured outputs = %g, %g, %g\n", computed_results[0], computed_results[1], computed_results[2]);
    if (Check_Prediction(computed_results, sample_classification)) {
        printf("Predicted output successfully\n");
    } else {
        printf("Failed to predict correct output\n");
    }
    fclose(pBiasFile);
    fclose(pWeightFile); 
    fclose(pAccuracyFile);  



    // Free memory
    Free_Memory(&h_neuron_bias,
        &h_neuron_input,
        &h_neuron_output,
        &h_neuron_delta,
        &h_neuron_forward_recieve_id,
        &h_neuron_forward_receive_start,
        &h_neuron_forward_recieve_weight,
        &h_layer_start,
        &h_layer_neuron_type);    

}
