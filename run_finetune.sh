export PATH=/usr/local/cuda/bin:$PATH
export DS_SKIP_CUDA_CHECK=1


version=$1
task=$2

# bash run_finetune.sh sft all

echo "train $version on $task"
deepspeed --include localhost:4,5,6,7 finetune.py \
            --data_file datasets/${version}/${task}.policy.train.json \
            --save_dir checkpoints/${version}_${task}_policy \
            --save_steps 50 \
            --batch_size 1 \
            --gradient_accumulation_steps 8 \
            --lr 1e-5 \
            --deepspeed --wandb



echo "train $version on $task"
deepspeed --include localhost:4,5,6,7 finetune.py \
            --data_file datasets/${version}/${task}.sft.train.json \
            --save_dir checkpoints/${version}_${task}_sft \
            --save_steps 100 \
            --max_epochs 1 \
            --batch_size 1 \
            --gradient_accumulation_steps 16 \
            --deepspeed --wandb
