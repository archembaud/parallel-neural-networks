#define NO_CHILDREN 30
#define SIGMA 0.75
#define FR 0.75

typedef struct {
    float *weights;
    float *bias;
    float fitness;
    float accuracy;
} GAChild;

void Allocate_GA_Memory(GAChild *children, GAChild *new_child, short NO_NEURONS, short NO_WEIGHTS, short NO_INPUTS);
void Free_GA_Memory(GAChild *children, GAChild new_child);
float Compute_GA_Fitness(float *delta, float *computed_results, float *sample_classification, short NO_NEURONS, short NO_OUTPUTS);
int Find_Fittest_Child(GAChild *children);
void Generate_New_Child(GAChild *children, GAChild new_Child, int fittest_parent, short NO_WEIGHTS, short NO_NEURONS);
void Update_Child(GAChild old_child, GAChild new_child, short NO_WEIGHTS, short NO_NEURONS);