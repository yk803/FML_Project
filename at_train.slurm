#!/bin/bash
#SBATCH --job-name=at_pgd_train
#SBATCH --open-mode=append
#SBATCH --output=%x_checkpoint_soar_%j.out
#SBATCH --error=%x_checkpoint_soar_%j.err
#SBATCH --export=ALL
#SBATCH --account=ds-ua-301-2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4


singularity exec --nv --bind /scratch/yy2949 --overlay /scratch/yy2949/overlay-25GB-500K.ext3:ro /scratch/yy2949/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
python /scratch/yy2949/fml2/train.py checkpoint_soar_last 10 1e-6 1e-1 5
"
