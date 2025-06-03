# Parallel Neural Networks

This repository contains a collection of neural network codes used for the purpose of research and development. It focuses on a range of topics:

* Numerical methods associated with neural networks, including minimization techniques, investigation into network design and activation functions,
* Acceleration using parallel computation, using OpenMP, explicit vectorization (AVX, SVE) and CUDA.

This is work in progress, and probably always will be, owing to the nature of research.

## Contents

* [CPU-iris](./CPU-iris/README.md): The prototype solver, designed to be simple and introduce students to neural networks.

* [GPU-iris](./GPU-iris/README.md): The prototype GPU solver, designed to be simple and introduce students to neural networks and the concept of data parallelism for a single GPU solver,
* [CPU-GA-iris](./CPU-GA-Iris/README.md): The modified CPU solver using a Differential Evolution Genetic Algoritm (DEGA) instead of the steepest descent method for updating values of weights and biases in the network.


## Prerequisites

* GCC is the compiler used in the various makefiles in the repo.
* For those interested in building and executing the SVE codes (in progress), you'll need the ARM C compiler installed.
* CUDA should be installed (i.e. nvcc compiler) for the GPU-accelerated neural network codes.
* All codes are prepared and tested within a Linux environment; WSL2 is an acceptable substitute for Windows users.

## Building and Running

A makefile is included in each of the folders; to make just navigate to the folder of interest and type:

```bash
make
```
and then run:

```bash
./main.exe
```

Typical code executions produce mention of the epoch and accuracy, together with one or more final tests. These are stochastic in nature - so your outputs will vary from ours, but they will typically look like this:

```bash
Epoch 996 of 1000 - accuracy = 0.953333
Epoch 997 of 1000 - accuracy = 0.953333
Epoch 998 of 1000 - accuracy = 0.953333
Epoch 999 of 1000 - accuracy = 0.953333
Checking prediction using randomly selected sample (146)
Sample classification = 0, 0, 1
Computed classification = 0.000157682, 0.265425, 0.900728
The classification of this sample was successful
```

## References

### Iris data set
* R. A. Fisher (1936). "The use of multiple measurements in taxonomic problems". Annals of Eugenics. 7 (2): 179–188. doi:10.1111/j.1469-1809.1936.tb02137.x. hdl:2440/15227.  
* Iris Data Set, UC Irvine Machine Learning Repository,
https://archive.ics.uci.edu/dataset/53/iris. Last visited on the 28th of May, 2025.

### Genetic Algorithms

* M. R. Smith, F.-A. Kuo, C.-W. Hsieh, J.-P. Yu, J.-S. Wu, and A. Ferguson, Rapid optimization of blast wave mitigation strategies using Quiet Direct Simulation and Genetic Algorithm, Computer Physics Communications, Volume 181, Issue 6,
2010,
Pages 1025-1036,
ISSN 0010-4655,
https://doi.org/10.1016/j.cpc.2010.02.009.

