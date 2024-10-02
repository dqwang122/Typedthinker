export PATH=$PATH:~/.local/bin



benchmark=$1 # logicqa
method=$2 # deductive
bz=$3 # 8

python sample_traj.py -i ${benchmark}.${method}.jsonl --max_num -1 --repeat_num 10 --batch_size ${bz}