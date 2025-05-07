#!/bin/bash
#SBATCH --job-name=opt
#SBATCH --output=opt.%j  
#SBATCH --error=opt.%j
#SBATCH --nodes=1   
#SBATCH --ntasks=28
#SBATCH --time=24:00:00 
#SBATCH --partition=loki4

echo "SLURM_NTASKS: $SLURM_NTASKS"

source ~/.bash_profile

if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
	echo "Error: SLURM_NTASKS is not set or is less than or equal to 0"
	exit 1
fi

pyte /data2_1/jinvk/cte/pyte/inp.yaml 
