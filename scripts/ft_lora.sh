export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage2.conf \
    open_instruct/finetune.py \
    --model_name_or_path /data/models/llama2/hf_llama2_7b \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --tokenizer_name /data/models/llama2/hf_llama2_7b \
    --use_slow_tokenizer \
    --train_file /data/whq/projects/svd_lora/open-instruct/data/alpaca_clean.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 200 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir output/alpaca_clean_original_lora_alpha64/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path /data/models/llama2/hf_llama2_7b \
    --lora_model_name_or_path output/alpaca_clean_original_lora_alpha64/ \
    --output_dir output/alpaca_clean_original_lora_merged_alpha64/ \
    --save_tokenizer
