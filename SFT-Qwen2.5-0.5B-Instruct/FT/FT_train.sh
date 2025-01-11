# --dataset "E:\\AI\\transformers-main\\examples\\pxd_practice\\Qwen2-Audio\\FT_data\\FT_train.jsonl" \
#--custom_train_dataset_path "E:\\AI\\transformers-main\\examples\\pxd_practice\\Qwen2-Audio\\FT_data\\FT_train.jsonl" \
#--custom_val_dataset_path  "E:\\AI\\transformers-main\\examples\\pxd_practice\\Qwen2-Audio\\FT_data\\FT_train.jsonl" \
# OMP_NUM_THREADS=4 NPROC_PER_NODE=1 
#--eval_strategy no \
#--max_length None \
#--flash_attention true \?
#--val_dataset "E:\\AI\\transformers-main\\examples\\pxd_practice\\Qwen2.5-0.5B-Instruct\\FT\\val_datasets_generated.jsonl" \
#训练完成后使用如下命令进行合并lora权重参数和原模型参数： python tools/merge_lora_weights_to_model.py --model_id_or_path /dir/to/your/base/model --model_revision master --ckpt_dir /dir/to/your/lora/model
#    python tools/merge_lora_weights_to_model.py --model_id_or_path "F:\\AI_model_save\\Qwen2.5-0.5B-Instruct"  --model_revision master --ckpt_dir r"output\v20-20241230-225736\checkpoint-10"
OMP_NUM_THREADS=4 NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model "F:\\AI_model_save\\Qwen2.5-0.5B-Instruct" \
    --model_type qwen2_5 \
    --train_type lora \
    --dataset "E:\\AI\\transformers-main\\examples\\pxd_practice\\Qwen2.5-0.5B-Instruct\\FT\\train_datasets_generated.jsonl" \
    --val_dataset "E:\\AI\\transformers-main\\examples\\pxd_practice\\Qwen2.5-0.5B-Instruct\\FT\\val_datasets_generated.jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author pxd \
    --model_name medical_sft
    