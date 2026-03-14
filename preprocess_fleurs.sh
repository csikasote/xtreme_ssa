#!/bin/sh

# This example submission script contains several important directives, please examine it thoroughly

# Do not put spaces between the start of the line and #SBATCH, the line must start exactly with #SBATCH, no spaces.
# Do not put spaces between the # and SBATCH

# The line below indicates which accounting group to log your job against
#SBATCH --account=compsci

# The line below selects the group of nodes you require
#SBATCH --partition=ada

# The line below reserves 1 worker node and 2 cores
#SBATCH --nodes=1 --ntasks=2

# The line below indicates the wall time your job will need, 10 hours for example.
#SBATCH --time=10:00:00

# A sensible name for your job, try to keep it short
#SBATCH --job-name="MyJob"

#Modify the lines below for email alerts. Valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL 
#SBATCH --mail-user=skscla001@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# The cluster is configured primarily for OpenMPI and PMI. Use srun to launch parallel jobs if your code is parallel aware.
# To protect the cluster from code that uses shared memory and grabs all available cores the cluster has the following 
# environment variable set by default: OMP_NUM_THREADS=1
# If you feel compelled to use OMP then uncomment the following line:
# export OMP_NUM_THREADS=$SLURM_NTASKS

# NB, for more information read https://computing.llnl.gov/linux/slurm/sbatch.html

# Use module to gain easy access to software, typing module avail lists all packages.
# Example:
# module load python/anaconda-python-3.7

# If your code is capable of running in parallel and requires a command line argument for the number of cores or threads such as -n 30 or -t 30 then you can link the reserved cores to this with the $SLURM_NTASKS variable for example -n $SLURM_NTASKS instead of -n 30

# Your science stuff goes here...
env_name="mms"
module load python/miniconda3-py3.12
source activate $env_name

LANGUAGES=("lg_ug" "sw_ke" "zu_za" "ig_ng" "sn_zw" "wo_sn" "kam_ke" "nso_za" "umb_ao" "ln_cd" "ff_sn")

ROOT_DIR="/scratch/skscla001/experiments/datasets/xtreme_ssa"

for lang_id in ${LANGUAGES[@]}; do
	echo "Processing $lang_id"

	mkdir -p "$ROOT_DIR/$lang_id"
	cd "$ROOT_DIR/$lang_id" || exit

	# download metadata
	wget -nc https://huggingface.co/datasets/google/fleurs/resolve/main/data/${lang_id}/audio/train.tar.gz
	wget -nc https://huggingface.co/datasets/google/fleurs/resolve/main/data/${lang_id}/audio/dev.tar.gz
	wget -nc https://huggingface.co/datasets/google/fleurs/resolve/main/data/${lang_id}/audio/test.tar.gz

	# extract audio
	tar -xzf train.tar.gz
	tar -xzf dev.tar.gz
	tar -xzf test.tar.gz

	# delete *.gz
	rm train.tar.gz
	rm dev.tar.gz
	rm test.tar.gz

	# download metadata
    	wget -nc https://huggingface.co/datasets/google/fleurs/raw/main/data/${lang_id}/train.tsv
    	wget -nc https://huggingface.co/datasets/google/fleurs/raw/main/data/${lang_id}/dev.tsv
    	wget -nc https://huggingface.co/datasets/google/fleurs/raw/main/data/${lang_id}/test.tsv

	# run preprocessing
	python "$ROOT_DIR/preprocess_fleurs.py" --lang="$lang_id" --root_dir="$ROOT_DIR"

	echo " "
done
