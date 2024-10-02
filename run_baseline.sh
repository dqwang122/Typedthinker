export PATH=$PATH:~/.local/bin


mode=$1 # zeroshot
repeat_num=$2 # 1
# gpu=$3 # 0
# CUDA_VISIBLE_DEVICES=${gpu} 

LOGIC_TASKS=(logicqa bbh)
MATH_TASKS=(gsm8k math)

MODEL=llama3_chat
# MODEL=mistral_chat

# bash run_baseline.sh zeroshot 1 60


for TASK in "${LOGIC_TASKS[@]}"
do
    echo "Evaluating ${MODEL}, ${TASK} under ${mode} with ${repeat_num} repeats"
    python baseline.py \
            --model ${MODEL} \
            --benchmark ${TASK} \
            --mode ${mode} \
            --problem logic \
            --load_prompt_from ${TASK}.icl.1.jsonl \
            --repeat_num ${repeat_num} \
            --batch_size 1000000 \
            --max_num -1 \
            --temperature 0.7
done

for TASK in "${MATH_TASKS[@]}"
do
    echo "Evaluating ${MODEL}, ${TASK} under ${mode}  with ${repeat_num} repeats"
    python baseline.py \
            --benchmark ${TASK} \
            --mode ${mode} \
            --problem math \
            --repeat_num ${repeat_num} \
            --load_prompt_from ${TASK}.icl.1.jsonl \
            --batch_size 1000000 \
            --max_num -1 \
            --temperature 0.7
done