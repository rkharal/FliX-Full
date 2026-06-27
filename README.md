# Coarse-Granular RTIndeX

This repository contains the code for the paper _More Bang For Your Buck(et): Fast and Space-efficient Hardware-accelerated Coarse-granular Indexing on GPUs_, preprint at <https://arxiv.org/abs/2406.03965>.

The code builds upon Ingo Wald's SIGGRAPH course, available at <https://github.com/ingowald/optix7course>.

The main CMake project file is located in the `src` directory.

## Project Structure

This project is organized into the following directories:

- **btree/**: Source code for a GPU B-Tree implementation.
- **ext/**: WarpCore dependencies and NVTX integration.
- **src/**: Source code for both coarse-granular and fine-granular index structures.
- **warpcore/**: Source code for WarpCore, a GPU Hash Table implementation.

## Directory Descriptions

- **btree/**: Contains the source code for the GPU B-Tree implementation by Muhammad Awad et al., based on the paper [_Engineering a High-Performance GPU B-Tree_](https://dl.acm.org/doi/pdf/10.1145/3293883.3295706). This code is included as a performance benchmark for comparisons with the coarse-granular index.

- **ext/**: Includes various WarpCore dependencies and NVIDIA’s NVTX (NVIDIA Tools Extension) integration. NVTX is a cross-platform API for annotating source code to provide contextual information to developer tools like Nsight Systems and Nsight Compute, in order to help with performance analysis by marking code sections, events, and threads.

- **src/**: The primary codebase for the project’s coarse-granular index structure, designed to leverage hardware-accelerated ray tracing cores for efficient indexing.

- **warpcore/**: Contains the source code for a GPU Hash Table implementation by Jünger et al., based on the paper [_WarpCore: Efficient Query Processing on Heterogeneous Hardware Accelerators_](https://arxiv.org/pdf/2009.07914). This is used to support hash-based indexing experiments and comparisons in the project.

## Prerequisites

- Your **GPU must support NVIDIA RTX (hardware raytracing acceleration)**. For consumer GPUs, this applies to **NVIDIA RTX 2000, 3000, and 4000** series, but there are workstation and data-center GPUs that also support RTX.
- Your GPU should have at least **24 GB of VRAM**. Smaller sizes will probably still be sufficient for most experiments, but are not guaranteed to.
- Ensure your system runs **NVIDIA's 555.42 GPU driver** or newer.
- Your system should have at least **32 GB of main memory**.
- All experiments are designed to run on **64-bit Linux**. We successfully tested **ArchLinux and AlmaLinux 9.4**.

## Installation

- Install the **CUDA 12.0 toolkit**:
  - Option 1: Through your package manager (if available in the required version).
  - Option 2: Manually by following the instructions on the NVIDIA website <https://developer.nvidia.com/cuda-downloads>. Depending on your current system setup, you might have to take additional steps, such as purging previous driver versions or disabling `nouveau`.
  - After the installation, test whether the CUDA toolkit is part of the `PATH` by running `nvcc --version`. If `nvcc` is not found, add the `bin` subdirectory of the CUDA toolkit (typically `/opt/cuda/bin` or `/usr/local/cuda/bin`) to your `PATH`.
- Parts of the code also depend on NVIDIA's **CUB and Thrust libraries**, which are included in the CUDA toolkit by default.
- Install **OptiX 8.0**:
  - Download OptiX from <https://developer.nvidia.com/designworks/optix/download> (requires an NVIDIA account).
  - Run the installer script, preferably in your home directory.
  - This will produce a directory called `NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64` or similar.
  - Set the environment variable `OptiX_INSTALL_DIR` to this directory, or add the directory to your `PATH`.
- Install **CMake 3.20** or later.
- Install **any version of `gcc`** that is compatible with your CUDA toolkit.
- If your system contains multiple GPUs, set the `CUDA_VISIBLE_DEVICES` variable to force execution on a specific one.

## Troubleshooting

- If CMake is unable to find CUDA, try setting the following variables:
  - `PATH=<CUDA toolkit root>/bin:$PATH`
  - `CUDA_HOME=<CUDA toolkit root>`
  - `CUDA_BIN_PATH=<CUDA toolkit root>/bin`
  - `CUDACXX=<CUDA toolkit root>/bin/nvcc`
- If you see something like `optix failed with code 7801`, update or re-install your driver.
- If you receive an error about unsupported GPU architectures, update or re-install your CUDA toolkit.
