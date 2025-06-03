#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ga_nn.h"

float weighted_max(float scale) {
    float rf = (float)rand()/RAND_MAX;
    if (rf < 0.5) {
        return scale*(1.0 - rf);
    }
    return scale*rf;
}

float generate_normal(float stddev) {
    // Generate two uniform random numbers between 0 and 1
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    // Apply the Box-Muller transform
    float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    // Scale to the desired standard deviation
    return z0 * stddev;
}


void Generate_New_Child(GAChild *children, GAChild new_Child, int fittest_parent, short NO_WEIGHTS, short NO_NEURONS) {
    // Need to allocate
    int parent_a = -1;
    int parent_b = -1;
    while ((parent_a == parent_b) || (parent_a == fittest_parent) || (parent_b == fittest_parent)) {
        parent_a = rand() % NO_CHILDREN;
        parent_b = rand() % NO_CHILDREN;
    }
    //printf("Selected parents %d, %d and %d\n", parent_a, parent_b, fittest_parent);
    // Now, we need to update the bias and weights on the new child
    for (short weight = 0; weight < NO_WEIGHTS; weight++) {
        float GF = weighted_max(FR);
        float RNSIG = weighted_max(SIGMA);
        new_Child.weights[weight] = GF*children[fittest_parent].weights[weight] + (1.0-GF)*children[parent_a].weights[weight]
                    + RNSIG*(children[parent_a].weights[weight] - children[parent_b].weights[weight]);
    }
    for (short neuron = 0; neuron < NO_NEURONS; neuron++) {
        float GF = weighted_max(FR);
        float RNSIG = weighted_max(SIGMA);
        new_Child.bias[neuron] = GF*children[fittest_parent].bias[neuron] + (1.0-GF)*children[parent_a].bias[neuron]
                    + RNSIG*(children[parent_a].bias[neuron] - children[parent_b].bias[neuron]);
    }
}

void Allocate_GA_Memory(GAChild *children, GAChild *new_child, short NO_NEURONS, short NO_WEIGHTS, short NO_INPUTS) {

    new_child->weights = (float*)malloc(NO_WEIGHTS*sizeof(float));
    new_child->bias = (float*)malloc(NO_NEURONS*sizeof(float));

    for (int i = 0; i < NO_CHILDREN; i++) {
        children[i].weights = (float*)malloc(NO_WEIGHTS*sizeof(float));
        children[i].bias = (float*)malloc(NO_NEURONS*sizeof(float));

        // While we are here, we can initalise the weights and bias values
        for (short neuron = 0; neuron < NO_NEURONS; neuron++) {
            if (neuron < NO_INPUTS) {
                children[i].bias[neuron] = 0.0; // No bias values on input neurons
            } else {
                children[i].bias[neuron] = (((float)rand() / RAND_MAX)-0.5)*2.0; // From -1 to 1    
            }
        }
        for (short weight = 0; weight < NO_WEIGHTS; weight++) {
            children[i].weights[weight] = (((float)rand()/ RAND_MAX)-0.5)*10.0; // From -5 to 5
        }
    }
}

void Free_GA_Memory(GAChild *children, GAChild new_child) {
    free(new_child.weights);
    free(new_child.bias);
    for (int i = 0; i < NO_CHILDREN; i++) {
        free(children[i].weights);
        free(children[i].bias);
    }
}

void Update_Child(GAChild old_child, GAChild new_child, short NO_WEIGHTS, short NO_NEURONS) {
    // Manually move these; its easier
    old_child.fitness = new_child.fitness;
    old_child.accuracy = new_child.accuracy;
    for (short weight = 0; weight < NO_WEIGHTS; weight++) {
        old_child.weights[weight] = new_child.weights[weight];
    } 
}


void Upload_GA_Properties(GAChild child, float *weights, float *bias, short NO_WEIGHTS, short NO_NEURONS) {
    // Move this child's properties into the network prior to measurement
    for (short neuron = 0; neuron < NO_NEURONS; neuron++) {
        bias[neuron] = child.bias[neuron];
    }
    for (short weight = 0; weight < NO_WEIGHTS; weight++) {
        weights[weight] = child.weights[weight];
    }
}


float Compute_GA_Fitness(float *delta, float *computed_results, float *sample_classification, short NO_NEURONS, short NO_OUTPUTS) {
    // Compute the fitness for this sample (these will sum over all samples)
    // We want the collective error (delta) on the outputs to matter as much as all of the deltas inside
    /*
    float weighted_sum_delta = 0.0;
    for (short output = 0; output < NO_OUTPUTS; output++) {
        weighted_sum_delta += fabs(delta[NO_NEURONS-output-1]);
    }
    // Now for the remainder - deltas on input and hidden nodes
    // Which are normalised by their number
    short valid_neurons = NO_NEURONS-NO_OUTPUTS;
    for (int neuron = 0; neuron < valid_neurons; neuron++) {
        weighted_sum_delta += (float)(1.0/valid_neurons)*fabs(delta[neuron]);
    }
    */
    float weighted_sum_delta = 0.0;
    // printf("-------------\n");
    for (short output = 0; output < NO_OUTPUTS; output++) {
        //printf("Expected = %g, Computed = %g\n", sample_classification[output], computed_results[output]);
        weighted_sum_delta += (computed_results[output] - sample_classification[output])*(computed_results[output] - sample_classification[output]);
    }
    //printf("Returning %g\n", weighted_sum_delta);
    return weighted_sum_delta;
}

int Find_Fittest_Child(GAChild *children) {
    // This is a minimization problem
    float best_fitness = 100000.0;
    int best_child = -1;
    for (int child = 0; child < NO_CHILDREN; child++) {
        // printf("Child %d has fitness %g and accuracy %g\n", child, children[child].fitness, children[child].accuracy);
        if (children[child].fitness < best_fitness) {
            best_child = child;
            best_fitness = children[child].fitness;
        }
    }
    printf("Best child seems to be: %d\n", best_child);
    return best_child;
}