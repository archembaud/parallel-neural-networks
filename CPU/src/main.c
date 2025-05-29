#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Total number of parameters is the number of weights + number of bias values (i.e neurons)

#define NO_SAMPLES 150
#define NO_LAYERS 3
#define NO_NEURONS 12
#define NO_WEIGHTS 35
#define NO_EPOCHS 400

// Data
float *data_input;
float *data_class;
const char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
const char *iris_data = "example/iris.data";

/*
Our network
                    4
0
                    5               9
1
                    6               10
2
                    7               11
3
                    8
Input             Hidden           Output

*/

// Layers
unsigned char layer_start[NO_LAYERS+1] = {0, 4, 9, 12};
unsigned char layer_neuron_id[NO_NEURONS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
unsigned char layer_neuron_type[NO_LAYERS] = {0, 1, 2}; // 0 = input, 1 = hidden, 2 = output

// Neurons
float neuron_bias[NO_NEURONS];                // Bias won't be applied to output layer
float neuron_input[NO_NEURONS];
float neuron_output[NO_NEURONS];
float neuron_delta[NO_NEURONS];              //

// Weights and Neuron Network
// For each neuron, keep track of its connecting neurons
// This can be stored in the same way CSR data is stored.
//                                                         |           |           |          |            |              |              |              | 
unsigned char neuron_forward_recieve_id[] =    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8};
//                                              0           4           8           12          16          20             25             30             35
unsigned char neuron_forward_receive_start[] = {0, 0, 0, 0, 0, 4, 8, 12, 16, 20, 25, 30, 35};
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


void Train_Network() {
    FILE *pBiasFile;
    pBiasFile = fopen("Bias.txt", "w");
    FILE *pWeightFile;
    pWeightFile = fopen("Weights.txt", "w");
    FILE *pAccuracyFile;
    pAccuracyFile = fopen("Accuracy.txt", "w");

    for (int epoch = 0; epoch < NO_EPOCHS; epoch++) {
        // Train our AI using all of the samples
        int no_successful_predictions = 0;
        for (int sample = 0; sample < NO_SAMPLES; sample++) {
            // This data has 4 inputs - 4 measurements of the plant - we can use for classification
            // Collect the input data for each sample
            float sample_input[4] = {data_input[sample*4+0], data_input[sample*4+1], data_input[sample*4+2], data_input[sample*4+3]};
            // Collect the classification data for each sample 
            float sample_classification[3] = {data_class[sample*3+0], data_class[sample*3+1], data_class[sample*3+2]};
            float computed_results[3];
            // Train the network using this sample
            Train_Network_using_single_sample(sample_input, sample_classification, computed_results, 0.1);
            no_successful_predictions += Check_Prediction(computed_results, sample_classification);
        }
        // Show training progress
        printf("Epoch %d of %d - accuracy = %g\n", epoch, NO_EPOCHS, (float)no_successful_predictions/NO_SAMPLES);
        fprintf(pAccuracyFile, "%d\t%g\n", epoch, (float)no_successful_predictions/NO_SAMPLES);
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

void Train_Network_using_single_sample(float *input, float *output, float *computed_output, float learning_rate) {

    for (int i = 0; i < NO_NEURONS; i++) {
        neuron_delta[i] = 0.0;
        neuron_input[i] = 0.0;
        neuron_output[i] = 0.0;
    }

    Run_Network(input, computed_output);

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

void Run_Network(float *input, float *computed_output) {
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



void Allocate_Memory() {
    data_input = (float*)malloc(sizeof(float) * NO_SAMPLES * 4);
    data_class = (float*)malloc(sizeof(float) * NO_SAMPLES * 3);
    // This data is allocated for in-execution analysis, which is currently lacking
    weight_history = (float*)malloc(sizeof(float)*NO_WEIGHTS*NO_EPOCHS);
    bias_history = (float*)malloc(sizeof(float)*NO_NEURONS*NO_EPOCHS);
    accuracy_history = (float*)malloc(sizeof(float)*NO_EPOCHS);
}


void load_data() {
    float p0, p1, p2, p3;
    char classification[30]; // 30 is long enough to fit the types
    float c0, c1, c2;
    int samples = 0;

    /* Load the iris data-set. */
    FILE *in = fopen(iris_data, "r");
    if (!in) {printf("Could not open file: %s\n", iris_data); exit(1); }

    // Grab the data from iris
    for (int sample = 0; sample < NO_SAMPLES; sample++) {
        fscanf(in, "%g,%g,%g,%g,%s\n", &p0, &p1, &p2, &p3, &classification);
        int index = sample*4;  // Four parameters per sample
        // Place these read values into our input array
        data_input[index+0] = p0;
        data_input[index+1] = p1;
        data_input[index+2] = p2;
        data_input[index+3] = p3;

        if (strcmp(classification, "Iris-setosa") == 0)     {c0 = 1.0;  c1 = 0.0; c2 = 0.0;}
        if (strcmp(classification, "Iris-versicolor") == 0) {c0 = 0.0;  c1 = 1.0; c2 = 0.0;}
        if (strcmp(classification, "Iris-virginica") == 0)  {c0 = 0.0;  c1 = 0.0; c2 = 1.0;}
        // Write these into our class array
        index = sample*3;
        data_class[index+0] = c0;
        data_class[index+1] = c1;
        data_class[index+2] = c2;
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
    int sample = rand() % 150;
    float computed_output[3]; // The output
    float sample_input[4] = {data_input[sample*4+0], data_input[sample*4+1], data_input[sample*4+2], data_input[sample*4+3]};
    float sample_classification[3] = {data_class[sample*3+0], data_class[sample*3+1], data_class[sample*3+2]};
    Run_Network(sample_input, computed_output);
    printf("Checking prediction using randomly selected sample (%d)\n", sample);
    printf("Sample classification = %g, %g, %g\n",sample_classification[0], sample_classification[1], sample_classification[2]);
    printf("Computed classification = %g, %g, %g\n",computed_output[0], computed_output[1], computed_output[2]);
    if (Check_Prediction(computed_output, sample_classification)) {
        printf("The classification of this sample was successful\n");
    } else {
        printf("The classification of this sample has failed\n");
    }

    Free_Memory();
}