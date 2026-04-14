#!/bin/bash

module load nvhpc
module load cuda
module load craype-accel-nvidia80

make cfd_euler_gpu cfd_euler_cpu

OUT=results.txt

echo "CPU vs GPU Performance Comparison - CFD Euler" > $OUT
echo "Date: $(date)" >> $OUT
echo "Nx_base=200  Ny_base=100  Steps=2000" >> $OUT
echo "----------------------------------------------" >> $OUT

export OMP_NUM_THREADS=8

for mult in 1 4 8 16; do
    NX=$((200 * mult))
    NY=$((100 * mult))
    echo "" | tee -a $OUT
    echo "Size ${mult}x  (Nx=${NX}, Ny=${NY})" | tee -a $OUT

    echo -n "  " | tee -a $OUT
    ./cfd_euler_cpu $mult CPU 2>&1 | tee -a $OUT

    echo -n "  " | tee -a $OUT
    ./cfd_euler_gpu $mult GPU 2>&1 | tee -a $OUT
done

echo "" >> $OUT
echo "Done." | tee -a $OUT
