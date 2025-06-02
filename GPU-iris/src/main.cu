#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "gpu_nn.h"

// Data - 4 inputs with 3 possible output classifications
const int NO_SAMPLES = 150;
float data_input[4*NO_SAMPLES]; 
float data_class[3*NO_SAMPLES];
const char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
const char *iris_data = "example/iris.data";

void load_data() {
    float p0, p1, p2, p3;
    char classification[30]; // 30 is long enough to fit the types
    float c0, c1, c2;
    int samples = 0;

    printf("Loading training data from Iris set\n");

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
    printf("Finished loading training data\n");
}


int main() {
    // Define out network - 4 input neurons, 5 neurons in a hidden layer, 3 output neurons 
    // -1 marks the end of the network
    short network[] = {4, 5, 3, -1};

    // Load data, store in data_input and data_class
    load_data();

    // Train using 1000 epochs with a learning rate of 0.1
    // This also prints a summary of the result
    Train_Network(data_input, sizeof(data_input), data_class, sizeof(data_class),  network, NO_SAMPLES, 1000, 0.1);

}