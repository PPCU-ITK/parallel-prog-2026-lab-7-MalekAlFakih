[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/JdNTJJYf)
# assignment-7

## Load modules

```bash
module load nvhpc
module load cuda
module load craype-accel-nvidia80
```

## Build

```bash
# build both GPU and CPU binaries for cfd_euler, plus laplace2d
make all

# or individually
make cfd_euler_gpu   # OpenMP GPU offload  (-mp=gpu -gpu=cc80 -Ofast)
make cfd_euler_cpu   # OpenMP CPU threads  (-mp -Ofast)
```

## Run laplace2d on the GPU

```bash
srun -p gpu --gres=gpu:1 --ntasks=1 --time=00:05:00 --mem=40G ./laplace2d
```

## Run CFD Euler CPU vs GPU comparison

The script `run_comparison.sh` runs both binaries at 1x, 4x, 8x, and 16x grid sizes
(Nx=200*mult, Ny=100*mult) and writes the timing results to `results.txt`.

```bash
srun -p gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --time=01:00:00 --mem=40G bash run_comparison.sh
```

Results are saved to `results.txt` in the working directory.

## Run a single size manually

```bash
# ./cfd_euler_gpu <size_multiplier> <label>
./cfd_euler_gpu 1  GPU    # default size  (Nx=200, Ny=100)
./cfd_euler_gpu 4  GPU    # 4x size       (Nx=800, Ny=400)
./cfd_euler_gpu 8  GPU    # 8x size       (Nx=1600, Ny=800)
./cfd_euler_gpu 16 GPU    # 16x size      (Nx=3200, Ny=1600)

./cfd_euler_cpu 1  CPU
./cfd_euler_cpu 4  CPU
```
