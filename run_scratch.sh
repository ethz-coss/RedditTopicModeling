#!/bin/bash
#SBATCH --job-name=scratch
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=40GB
#SBATCH --time=00:20:00
#SBATCH --output=output_scratch.txt
#SBATCH --error=error_scratch.txt

module load gcc/8.2.0 python_gpu/3.11.2 cuda/11.2.2

source reddit_env/bin/activate

which python

python ./RedditProject/TransformerEmbeddings/wrappers.py





