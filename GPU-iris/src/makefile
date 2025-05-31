all: GPU CPU 
	nvcc main.o gpu_nn.o -code=sm_80 -arch=compute_80 -O3 -o main.exe
CPU:
	nvcc main.cu -c -O3
GPU:
	nvcc gpu_nn.cu -c -code=sm_80 -arch=compute_80 -O3
