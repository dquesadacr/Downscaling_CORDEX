#!/bin/bash

# Written by Dánnell Quesada

#=======================================================
# SBATCH options
#=======================================================
#SBATCH -c 6 #number of cores
#SBATCH --time=24:00:00
##SBATCH --time=12:00:00
#SBATCH --mem=330GB
##SBATCH --mem=240GB
#SBATCH --nodes=1
#SBATCH -p alpha
## SBATCH -w taurusi80[03-10] # Select specific node, if desired
##SBATCH -o /beegfs/ws/1/s9941460-Proj/jobs/proj_sud.out
##SBATCH -e /beegfs/ws/1/s9941460-Proj/jobs/proj_sud.err
##SBATCH -J Proj_sud
##SBATCH --exclude=taurusi80[26-34]
#SBATCH --gres=gpu:1
#SBATCH --mail-user dannell.quesada@tu-dresden.de
#SBATCH --mail-type END

module --force purge
nvidia-modprobe -u -c=0
cp proj_rcms.R V-"$1"/
cp proj_sing_run.sh V-"$1"/

singularity exec --nv -B V-"$1"/:/data -B /beegfs/ws/1/s9941460-EUR11/full_fill/rds/:/data/rds Rep_Proj.sif bash /data/proj_sing_run.sh $1 $2
