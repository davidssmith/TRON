CC=nvcc
CFLAGS=-O3 -Wno-deprecated-gpu-targets --use_fast_math -D_FORCE_INLINES -DCUDA_HOST_MALLOC
LFLAGS=-lcufft

# -DBEATTY_BETA     to use Beatty et al. kernel shape parameter

objects = tron.o ra.o float16.o

all: tron

tron: $(objects)
	$(CC) $(LFLAGS) $(objects) -o tron

%.o: %.cu
	$(CC) $(CFLAGS) -dc $< -o $@

clean:
	rm -f *.o tron
