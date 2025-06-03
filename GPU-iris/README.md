# GPU (Prototype)

This solver is the GPU extended prototype of the CPU prototype.

This method leverages GPU through the use of data parallelism - that is, in each epoch, each sample used for training the network (i.e. computing updated values of weights and biases using backpropagation and the steepest descent method) uses a single CUDA thread. Hence, the degeee of parallelism increases as the number of training samples increases.

In this case, we are looking at the Iris data set - which includes 150 samples. In this case, we use the minimum sensible number of threads per block (32), resulting in the parallelisation across 4 meager blocks.

Each thread maintains its own values for neuron deltas, neuron inputs and outputs - these are then used (in a classical reduction manner) to update the weights and bias values, using atomic add operations (i.e. slow).

## Details

The neural network described with this example contains 12 neurons (numbered 0 to 11) connected by a network with 35 weights applied to said connections, as seen in the CPU demo:

There are:

* 4 input nodes (0-3), and
* 3 output nodes (nodes 9, 10 and 11).
* a single hidden layer with 5 nodes (4-8).

The data used for this demonstration is the Iris data set, based on data collected by Fisher in 1936. This set has been used for decades to test classification and data analysis techniques. (References on the main README file)

The Iris data set has 4 continuous input variables:
* sepal length,
* sepal width,
* petal length, and
* petal width.

Provided with each of these is a classification: Iris Setosa, Iris Versicolour, or Iris Virginica.

### Warning! ###

In spectactulary bad form, this demonstration loads the entire set and uses it for training - after which it randomly selects one of the data points and checks if the classification was successful. Normally, one might expect that a fraction of the sample data set is exempted from the training set for proper testing.

## Running the code

**Note**: I am assuming you have correctly installed Nvidia's CUDA toolkit and the nvcc compiler.

Navigate to the src directory and run:

```bash
make && ./main.exe
```

Running this code generates 3 files:

* Accuracy.txt - contains the training accuracy vs epoch number,
* Weights.txt - contains the weights on the network, and
* Bias.txt - contains the bias values of the neurons.

## Expected Output

The code runs in a very similar way to the CPU, and produces an output like this:

```bash
------- SAMPLE 148 ---------
Checking input:6.2 3.4 5.4 2.3
Checking Outputs: 0 0 1
Neuron Inputs: 6.2, 3.4, 5.4, 2.3, 12.303, 11.9042, 5.59745, 11.8525, 12.4193, -3.83137, -1.14926, -1.72545,
Neuron Outputs: 6.2, 3.4, 5.4, 2.3, 0.999992, 0.999985, 0.995636, 0.999986, 0.99999, 0.0147546, 0.496696, 0.578105,
Neuron Deltas: -0.000469943, -0.00109494, 0.00120744, 0.00113743, -0.0247696, -0.0250331, 0.106499, -0.0139625, -0.0339408, -0.000214485, -0.124169, 0.1029,
------- SAMPLE 149 ---------
Checking input:5.9 3 5.1 1.8
Checking Outputs: 0 0 1
Neuron Inputs: 5.9, 3, 5.1, 1.8, 11.2615, 11.2275, 4.83456, 11.0579, 11.2491, -3.79792, -1.15891, -1.75831,
Neuron Outputs: 5.9, 3, 5.1, 1.8, 0.999977, 0.99997, 0.990688, 0.99997, 0.999966, 0.0152488, 0.494283, 0.57007,
Neuron Deltas: -0.00104655, -0.00243794, 0.00268779, 0.00253154, -0.0258731, -0.0261196, 0.111659, -0.0153685, -0.0352543, -0.000228981, -0.123555, 0.105372,
=======================================

Neuron Bias: 0, 0, 0, 0, 0.565769, 0.827722, 0.177866, 0.655271, 0.952992, 0.363482, -1.13249, -1.9992,
Weights: 0.748741 0.674909 0.568312 0.752883 0.868226 0.256825 0.840392 0.122801 -0.961329 -2.33875 2.63549 2.46816 0.716702 0.442823 0.845033 0.297864 0.825719 0.984906 0.141082 0.972392 0.815488 0.880471 -6.77284 0.0625979 0.75373 -0.360338 -0.337 1.93972 -0.943412 -0.307185 -1.39566 -1.37729 6.7125 -1.68631 -1.69535
```