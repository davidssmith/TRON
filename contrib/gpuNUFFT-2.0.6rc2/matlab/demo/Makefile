CC=nvcc
# -arch=sm_20
CFLAGS=-O3 --use_fast_math -Xptxas -v -DCUDA_HOST_MALLOC
LFLAGS=-lcufft

objects = gpurecon.o

all: gpurecon

gpurecon: $(objects)
	$(CC) $(LFLAGS) $(objects) -o gpurecon

%.o: %.cu
	$(CC) $(CFLAGS) -dc $< -o $@

clean:
	rm -f *.o gpurecon
