CC=nvcc
CFLAGS=-O3 --use_fast_math -DCUDA_HOST_MALLOC #-DKERN_KB
LFLAGS=-lcufft

objects = rr2d.o ra.o

all: rr2d

rr2d: $(objects)
	$(CC) $(LFLAGS) $(objects) -o rr2d

%.o: %.cu
	$(CC) $(CFLAGS) -dc $< -o $@

clean:
	rm -f *.o rr2d
