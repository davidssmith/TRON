# TRON

Trajectory Optimized Nufft: Faster Non-Cartesian MRI Reconstruction through
Prior Knowledge and Parallel Architectures

Fast non-Cartesian MRI reconstruction (“gridding”) requires interpolation of
non-uniformly sampled Fourier data onto a Cartesian grid. This interpolation
is slow due to complicated, non-local data access patterns that are hard to
optimize for modern, parallel CPU architectures.

TRON is a gridding code customized for a standard radial (both linear and
golden angle) trajectory and Nvidia’s CUDA architecture to eliminate the
bottlenecks that gridding algorithms encounter when written for arbitrary
trajectories and hardware.

TRON is 30x faster than gpuNUFFT (a fast GPU code) and 75x faster than the
image reconstruction toolbox (IRT) of Fessler (a ubiquitous CPU-based Matlab
code). TRON eliminates the need to presort data or perform a separate sample
density compensation, while  comprising 50% less code than the minimal subset
of IRT tested.
