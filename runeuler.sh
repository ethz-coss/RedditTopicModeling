#!/bin/bash
#SBATCH --job-name=my_job_name
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=100MB
#SBATCH --time=00:01:00
#SBATCH --output=output.txt
#SBATCH --error=error.txt

module load gcc/8.2.0 python_gpu/3.11.2

pip install duckdb

python try_batch.py
