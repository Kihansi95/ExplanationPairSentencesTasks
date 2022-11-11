#!/bin/bash
#
#SBATCH --job-name=lstm_attention_esnli
#SBATCH --output=%j.out.log # output file (%j = job ID)
#SBATCH --error=%j.err.log # error file (%j = job ID)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=duc-hau.nguyen@irisa.fr
#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH -C v100-32g
#SBATCH --gres=gpu:6

conda activate cuda113
EXEC_FILE=src/lstm_attention.py
echo
echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at `date +"%T, %d-%m-%Y"`

python $EXEC_FILE -o $RUNDIR -b 1024 -e 50 --vectors glove.840B.300d -m exp --name lstm_attention_esnli --data esnli --lambda_supervise 0.0 --n_lstm 1 --version run=0_lstm=1_lsup=0.0 --strategy ddp

echo Done