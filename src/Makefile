CC=nvcc
CFLAGS=-O3 --use_fast_math -D_FORCE_INLINES -DKERN_KB -DCUDA_HOST_MALLOC  -DWALSH_COMB
LFLAGS=-lcufft

objects = tron.o ra.o

all: tron

tron: $(objects)
	$(CC) $(LFLAGS) $(objects) -o tron

%.o: %.cu
	$(CC) $(CFLAGS) -dc $< -o $@

clean:
	rm -f *.o tron
