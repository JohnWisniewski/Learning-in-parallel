CC=nvcc
#CFLAGS=-g
CFLAGS=

# Compile all CUDA files together so kernels are visible
nnp: kernels.cu nnp.cu loader.cc main.cc loader.h config.h nnp.h kernels.h
	$(CC) $(CFLAGS) kernels.cu nnp.cu loader.cc main.cc -o nnp -lm

clean: prod
	rm -f nnp 
prod:
	rm -f *.o model.bin
