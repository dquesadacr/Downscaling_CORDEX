#!/bin/bash

# Written by Dánnell Quesada

mkdir -p jobs

echo iter=$1 part=$2 
sbatch -J Proj_sud_"$1"-"$2" -o $(pwd)/jobs/Proj_sud_"$1"-"$2".out -e $(pwd)/jobs/Proj_sud_"$1"-"$2".err job_proj.sh $1 $2
