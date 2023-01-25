#!/bin/bash

# Written by Dánnell Quesada

#=======================================================
# SBATCH options
#=======================================================

#SBATCH -c 6 #number of cores
#SBATCH --time=24:00:00
#SBATCH --mem=120GB
#SBATCH --nodes=1
#SBATCH -J harm_sing
#SBATCH -o /beegfs/ws/1/s9941460-EUR11/Jobs/harm_sing.out
#SBATCH -e /beegfs/ws/1/s9941460-EUR11/Jobs/harm_sing.err
#SBATCH --mail-user dannell.quesada@tu-dresden.de
#SBATCH --mail-type END

module --force purge

singularity exec -B $(pwd)/full_fill:/data Rep_Proj.sif bash /data/rds/sing_harm_norm.sh

