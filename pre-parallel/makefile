CC=nvcc
#CFLAGS=-g
CFLAGS=

# Kernels are now in kernels.h, so we only need to compile nnp.cu
nnp: nnp.cu loader.cc main.cc loader.h config.h nnp.h kernels.h
	$(CC) $(CFLAGS) nnp.cu loader.cc main.cc -o nnp -lm

clean: prod
	rm -f nnp 
prod:
	rm -f *.o model.bin
