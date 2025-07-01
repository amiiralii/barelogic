#!/bin/tcsh


bsub << EOF
#!/bin/tcsh
#BSUB -W 4520
#BSUB -n 1
#BSUB -J R
#BSUB -o ./Logfile/%J.out
#BSUB -e ./Logfile/%J.err
conda activate /usr/local/usrapps/tjmenzie/arayega/my_env
python extend4.py ../../moot/auto93.csv | tee results/auto93.csv
conda deactivate
EOF
