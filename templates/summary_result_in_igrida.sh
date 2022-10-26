#!/usr/bin/env bash
#OAR -n summary_result
#OAR -l core=1,walltime=24:00:00
#OAR -O /srv/tempdd/dunguyen/RUNS/OARLOG/%jobid%.out.log
#OAR -E /srv/tempdd/dunguyen/RUNS/OARLOG/%jobid%.err.log

conda deactivate
source $VENV/eps/bin/activate
EXEC_FILE=src/summarize_result.py
echo
echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at `date +"%T, %d-%m-%Y"`
python $EXEC_FILE --log_dir $RUNDIR/logs --out_dir $RUNDIR --round 4 --experiment lstm_attention_hatexplain_heuristic lstm_attention_yelp50_heuristic

echo Done