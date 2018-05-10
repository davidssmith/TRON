# TRON: TRajectory Optimized Nufft

## Introduction

This is the repository for the paper, "Trajectory Optimized Nufft: Faster Non-Cartesian MRI Reconstruction through
Prior Knowledge and Parallel Architectures".

Fast non-Cartesian MRI reconstruction ("gridding") requires interpolation of
non-uniformly sampled Fourier data onto a Cartesian grid. This interpolation
is slow due to complicated, non-local data access patterns that are hard to
optimize for modern, parallel CPU architectures.

TRON is a gridding code customized for a standard radial (both linear and
golden angle) trajectory and Nvidia's CUDA architecture to eliminate the
bottlenecks that gridding algorithms encounter when written for arbitrary
trajectories and hardware.

TRON is 30x faster than gpuNUFFT (a fast GPU code) and 75x faster than the
image reconstruction toolbox (IRT) of Fessler (a ubiquitous CPU-based Matlab
code). TRON eliminates the need to presort data or perform a separate sample
density compensation, while  comprising 50% less code than the minimal subset
of IRT tested.

## Reproducing the Paper Results

To get generate the paper figures and data, follow these steps:

Requirements: Newish version of MATLAB and CUDA.


0. Make sure you have a working CUDA installation with cuFFT and the compiler `nvcc` in
   your path.

1. Clone and compile [gpuNUFFT](https://github.com/andyschwarzl/gpuNUFFT) into your folder of choice.

2. Clone and compile [BART](https://github.com/mrirecon/bart) into your folder of choice.

3. Correct the path to gpuNUFFT and BART in the RUNME2 script.

4. Run `src/RUNME1_tron_degrid_phantom.sh` on the command line.

5. Change to the `src/` subdirectory and run `src/RUNME2_others_degrid_phantom.m` 
   in MATLAB. This should not take long to finish.

6. Go back to the command line and run `src/RUNME3_tron_grid_all.sh`. This will
   also not take long.

7. Go back to MATLAB and run `./RUNME4_others_grid_all.m`. This is slow. An hour
   is normal. Put your feet up and read our paper while you wait.

8. If all steps run without errors, you're done. Congratulations! The
   paper figures should be in `src/figs` and the reconstructions will be in 
   `src/output`.
