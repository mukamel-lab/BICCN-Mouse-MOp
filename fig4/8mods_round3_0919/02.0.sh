nohup find * -name "config*.py" | xargs -P 2 -I {} time /cndd/fangming/CEMBA/snmcseq_dev/SCF_notebook_routine.py -d ./ -c {} > log_{}.log 2>&1 &
