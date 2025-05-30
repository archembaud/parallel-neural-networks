#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define NO_SAMPLES 4177
#define NO_LAYERS 3
#define NO_NEURONS 24
#define NO_WEIGHTS 125
#define NO_EPOCHS 1000000

// Data
float *data_input;
float *data_class;
const char *abalone_data = "example/abalone.data";

/*
Our network
                    8
0
                    9               
1
                    10               23
2
                    11               
                    ...

7
                    22
Input             Hidden           Output (Single)

*/

// Layers
unsigned char layer_start[NO_LAYERS+1] = {0, 8, 23, 24};
unsigned char layer_neuron_type[NO_LAYERS] = {0, 1, 2}; // 0 = input, 1 = hidden, 2 = output

// Neurons
float neuron_bias[NO_NEURONS];                // Bias won't be applied to output layer
float neuron_input[NO_NEURONS];
float neuron_output[NO_NEURONS];
float neuron_delta[NO_NEURONS];              //

// Weights and Neuron Network
// For each neuron, keep track of its connecting neurons
// This can be stored in the same way CSR data is stored.
unsigned char neuron_forward_recieve_id[] =    {0, 1, 2, 3, 4, 5, 6, 7,  // 0, Ending on 8
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 1, 16
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 2 ,24
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 3, 32
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 4, 40
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 5, 48
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 6, 56
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 7, 64
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 8, 72
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 9, 80
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 10, 88
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 11, 96
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 12, 104
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 13, 112
                                                0, 1, 2, 3, 4, 5, 6, 7,  // 14, 120
                                                8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}; // Single output layer, 135 

unsigned char neuron_forward_receive_start[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 135};
float neuron_forward_recieve_weight[NO_WEIGHTS];

// Let's save some advanced results for graphing and analysis
float *weight_history;
float *bias_history;
float *accuracy_history;

float Activation_Function(float input) {
    // Use the sigmoid function
    return (1.0/(1.0 + expf(-input)));
}


void Init() {
    // Randomly set bias values
    srand(time(NULL));
    for (int i = 0; i < NO_NEURONS; i++) {
        neuron_bias[i] = (float)rand()/RAND_MAX;
    }
    // Randomly set the weights
    for (int i = 0; i < NO_WEIGHTS; i++) {
        neuron_forward_recieve_weight[i] = (float)rand()/RAND_MAX;
    }
}

float Check_Prediction(float computed_ring_number, float expected_ring_number) {
    // Sum of error squared
    return ( (computed_ring_number - expected_ring_number)*(computed_ring_number - expected_ring_number) );
}


float Run_Network(float *input) {
    // Reset our deltas
    for (int i = 0; i < NO_NEURONS; i++) {
        neuron_delta[i] = 0.0;
        neuron_input[i] = 0.0;
        neuron_output[i] = 0.0;
    }

    // Moving forward
    for (int layer = 0; layer < NO_LAYERS; layer++) {
        //printf("Processing layer %d\n", layer);
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
                    //printf("Input Neuron %d has an output of %g\n", neuron_id, neuron_output[neuron_id]);
                } else {
                    printf("ERROR: neuron %d has no inputs\n", neuron_id);
                }
            } else {
                // We have a middle layer, or an output layer.
                float neuron_input_sum = 0.0;
                for (int source = 0; source < no_sources; source++) {
                    int source_id = neuron_forward_recieve_id[neuron_forward_receive_start[neuron_id] + source];
                    //printf("Neuron %d has a source neuron %d with current output %g\n", neuron_id, source_id, neuron_output[source_id]);
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

    // Now to return the output vector; there is a single node here for this problem
    return neuron_output[NO_NEURONS-1];
}


float Train_Network_using_single_sample(float *input, float output, float learning_rate) {

    for (int i = 0; i < NO_NEURONS; i++) {
        neuron_delta[i] = 0.0;
        neuron_input[i] = 0.0;
        neuron_output[i] = 0.0;
    }

    float computed_output = Run_Network(input); // Computed output - the number of rings - is a single value

    // Now we need to run it backwards (backwards propagation)
    // Start by computing the deltas at the output layer.
    // So our outer loop is run in reverse
    for (int layer = (NO_LAYERS-1); layer >= 0; layer--) {
        for (int layer_neuron = 0; layer_neuron < (layer_start[layer+1] - layer_start[layer]); layer_neuron++) {
            int neuron_id = layer_start[layer] + layer_neuron;
            if (layer_neuron_type[layer] == 2) {
                // This is the output layer; there is only one neuron in the output layer in this case.
                // There are many possible different methods for computing this delta
                // Sigmoid Activation approach
                neuron_delta[neuron_id] = (output - computed_output)
                *computed_output*(1.0 - computed_output);

                // printf("Computed delta on output node %d = %g. Output = %g, Computed output = %g\n", neuron_id, neuron_delta[neuron_id], output, computed_output);
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
    // Return the computed output
    return computed_output;
}


void Train_Network() {
    FILE *pBiasFile;
    pBiasFile = fopen("Bias.txt", "w");
    FILE *pWeightFile;
    pWeightFile = fopen("Weights.txt", "w");
    FILE *pAccuracyFile;
    pAccuracyFile = fopen("Accuracy.txt", "w");

    for (int epoch = 0; epoch < NO_EPOCHS; epoch++) {
        // Train our AI using all of the samples
        float cummulative_error = 0;
        for (int sample = 0; sample < NO_SAMPLES; sample++) {
            // This data has 8 inputs - 8 measurements taken by Nash et. al. (1994)
            /*
            Nash, W., Sellers, T., Talbot, S., Cawthorn, A., & Ford, W. (1994).
            Abalone [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55C7W.
            */
            // Collect the input data for each sample
            float sample_input[8] = {data_input[sample*8+0], data_input[sample*8+1], data_input[sample*8+2], data_input[sample*8+3],
                                     data_input[sample*8+4], data_input[sample*8+5], data_input[sample*8+6], data_input[sample*8+7]};

            // Collect the classification data for each sample 
            float sample_age = data_class[sample];
            float computed_result = Train_Network_using_single_sample(sample_input, sample_age, 0.5);
            cummulative_error += Check_Prediction(computed_result, sample_age);
        }
        // Show training progress
        printf("Epoch %d of %d - sum of error squared = %g\n", epoch, NO_EPOCHS, (float)cummulative_error/NO_SAMPLES);
        fprintf(pAccuracyFile, "%d\t%g\n", epoch, (float)cummulative_error/NO_SAMPLES);
        // Now I want to record the weights and the bias values
        for (int i = 0; i < NO_NEURONS; i++) {
            bias_history[epoch*NO_NEURONS + i] = neuron_bias[i];
            fprintf(pBiasFile, "%d\t%d\t%g\n",epoch, i, neuron_bias[i]);
        }
        for (int i = 0; i < NO_WEIGHTS; i++) {
            weight_history[epoch*NO_WEIGHTS + i] = neuron_forward_recieve_weight[i];
            fprintf(pWeightFile, "%d\t%d\t%g\n",epoch, i, neuron_forward_recieve_weight[i]);
        }
    }
    fclose(pBiasFile);
    fclose(pWeightFile); 
    fclose(pAccuracyFile);   
}


void Allocate_Memory() {
    data_input = (float*)malloc(sizeof(float) * NO_SAMPLES * 8);   // 8 features on this data set
    data_class = (float*)malloc(sizeof(float) * NO_SAMPLES);       // The number of rings is the output
    // This data is allocated for in-execution analysis, which is currently lacking
    weight_history = (float*)malloc(sizeof(float)*NO_WEIGHTS*NO_EPOCHS);
    bias_history = (float*)malloc(sizeof(float)*NO_NEURONS*NO_EPOCHS);
    accuracy_history = (float*)malloc(sizeof(float)*NO_EPOCHS);
}

typedef struct {
    char gender;
    float values[8];
} Data;


void load_data() {
    printf("Loading data in\n");
    /* Load the Abalone data-set. */
    FILE *in = fopen(abalone_data, "r");
    if (!in) {printf("Could not open file: %s\n", abalone_data); exit(1); }

    char line[200];
    int count = 0;

    while (count < NO_SAMPLES && fgets(line, sizeof(line), in)) {
        int index = count*8;
        char *token;
        // Get the first token
        token = strtok(line, ",");
        // Iterate through the string, printing each token
        int line_count = 0;
        while (token != NULL) {
            if (line_count == 0) {
                // This is the sex of the abalone
                if (strcmp(token, "M") == 0) {
                    data_input[index] = -1.0;
                } else if (strcmp(token, "F") == 0) {
                    data_input[index] = 1.0;
                } else {
                    data_input[index] = 0.0;
                }
            } else if (line_count == 8) {
                // This is the classification
                data_class[count] = (float)atoi(token)/10.0; // Normalise the output
            } else {
                data_input[index+line_count] = atof(token);
            }
            line_count++;
            token = strtok(NULL, ",");
        }
        /*
        printf("Final data input: %g, %g, %g, %g, %g, %g, %g, %g\n",
            data_input[index], data_input[index+1], data_input[index+2], data_input[index+3],
            data_input[index+4], data_input[index+5], data_input[index+6], data_input[index+7]);
        printf("Final classification = %g\n", data_class[count]);
        */
        count++;
    }

    fclose(in);
}

void Free_Memory() {
    free(data_input);
    free(data_class);
    free(weight_history);
    free(bias_history);
}



int main() {
    Allocate_Memory();
    Init();
    load_data();
    Train_Network();

    // Now pick one of the samples and test it
    int sample = rand() % NO_SAMPLES;
    float computed_output; // The output
    float sample_input[8] = {data_input[sample*8+0], data_input[sample*8+1], data_input[sample*8+2], data_input[sample*8+3],
                             data_input[sample*8+4], data_input[sample*8+5], data_input[sample*8+6], data_input[sample*8+7]};
    float sample_output = data_class[sample]; // The number of rings
    computed_output = Run_Network(sample_input);
    printf("Checking prediction using randomly selected sample (%d)\n", sample);
    printf("Sample number of rings = %g\n",sample_output);
    printf("Computed number of rings = %g\n",computed_output);
    float error = Check_Prediction(computed_output, sample_output);
    printf("Error squared = %g\n", error);
    Free_Memory();
}