all: MAIN SERIAL_NN GA
	gcc main.o serial_nn.o ga_nn.o -O3 -o main.exe -lm
MAIN:
	gcc main.c -O3 -c -lm
SERIAL_NN:
	gcc serial_nn.c -O3 -c -lm
GA:
	gcc ga_nn.c -O3 -c
