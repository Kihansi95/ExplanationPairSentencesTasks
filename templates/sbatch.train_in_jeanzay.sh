#!/usr/bin/env bash
#SBATCH --job-name=mlp_attention
#SBATCH --output=/gpfswork/rech/tme/.../JZLOGS/%j.out.log # output file (%j = job ID)
#SBATCH --error=/gpfswork/rech/tme/.../JZLOGS/%j.err.log # error file (%j = job ID)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=duc-hau.nguyen@irisa.fr
#SBATCH --nodes=1             # reserve 1 node
#SBATCH --ntasks=4            # reserve 4 tasks (or processes)
#SBATCH --cpus-per-task=10    # reserve 10 CPUs per task (and associated memory)
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH -C v100-32g
#SBATCH --gres=gpu:4
source ~/.bashrc
conda activate cuda113
EXEC_FILE=src/mlp_attention.py
echo
echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at `date +"%T, %d-%m-%Y"`

srun python $EXEC_FILE -o $RUNDIR \
                      --vectors glove.840B.300d \
                      --devices $SLURM_GPUS_ON_NODE \
                      --num_nodes $SLURM_NNODES \
                      --mode exp \
                      --batch_size 512 \
                      --epoch 50 \
                      --n_kernel 1 \
                      --concat_context \
                      --name $SLURM_JOB_NAME \
                      --version yelphat50/run=0_ncontext=1_concat=1 \
                      --data yelphat50 \
                      --strategy ddp_find_off

echo Done