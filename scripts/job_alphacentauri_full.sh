#!/bin/bash

# Written by Dánnell Quesada

#=======================================================
# SBATCH options
#=======================================================
#SBATCH -c 8 #number of cores
#SBATCH --time=24:00:00
#SBATCH --mem=300GB
#SBATCH --nodes=1
#SBATCH -p alpha
## SBATCH -w taurusi80[03-10] # Select specific node, if desired
##SBATCH --exclude=taurusi80[10-34]
##SBATCH --exclude=taurusi80[03-20,30-34]
#SBATCH --exclude=taurusi80[25-34]
#SBATCH --gres=gpu:1
#SBATCH --mail-user dannell.quesada@tu-dresden.de
#SBATCH --mail-type END

module --force purge
mkdir -p V-"$1"/Data/all V-"$1"/models/all

cp -n Data/all/y_sud.rds V-"$1"/Data/all/
cp -n Data/all/x_32.rds V-"$1"/Data/all/
cp -n sing_run.sh V-"$1"/
cp -n train_scripts/Sud_cnn_$1.R V-"$1"/
cp -n parse_aux/unet_def.R V-"$1"/
cp -n parse_aux/aux_funs_train.R V-"$1"/

nvidia-modprobe -u -c=0
singularity exec --nv -B V-"$1"/:/data Rep_Proj.sif bash /data/sing_run.sh $1 $2 $3 $4 $5

# To analyse the validation results easier among different iterations
mkdir -p val_hist/V-"$1"/$4/$3/
cp V-"$1"/Data/all/$4/$3/validation_CNN* val_hist/V-"$1"/$4/$3/
cp V-"$1"/Data/all/$4/$3/hist_train_CNN* val_hist/V-"$1"/$4/$3/
