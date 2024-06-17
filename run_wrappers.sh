#!/bin/bash
#SBATCH --job-name=filtered_pipeline
#SBATCH --nodes=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=50GB
#SBATCH --time=01:30:00
#SBATCH --output=output_wrap.txt
#SBATCH --error=error_wrap.txt

module load gcc/8.2.0 python_gpu/3.11.2 cuda/11.2.2

source reddit_env/bin/activate

which python

python ./RedditProject/TransformerEmbeddings/wrappers.py





