#!/bin/bash

device=$1

if [ $device = "cpu" ];then
  srun --nodes=1 --ntasks-per-node 1 --cpus-per-task 128 --mem 250GB -p standard --pty bash -i
elif [ $device = "gpu" ];then
  srun --job-name=test --nodes=1 --ntasks-per-node 1 --cpus-per-task 16 --mem 500GB --gres=gpu:4 -p gpumid --reservation=training --pty bash -i
  # srun --job-name=test --nodes=1 --ntasks-per-node 1 --cpus-per-task 16 --mem 500GB --gres=gpu:4 -p gpuhigh --pty bash -i 
fi

