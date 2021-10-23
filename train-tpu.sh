export XLA_USE_BF16=1
#export PT_XLA_DEBUG=1
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
python3 ./examples/pytorch/xla_spawn.py --num_cores=8 \
	 examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-3b \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /mnt/common/t5-3b-checkpoints \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --pad_to_max_length \
    --save_total_limit 5 \
    --save_steps 1000 \
    --logging_steps 100 \
    --resume_from_checkpoint /mnt/common/t5-3b-checkpoints/checkpoint-10000
    
    #--resume_from_checkpoint /tmp/tst-summarization/checkpoint-1000 \
    
#--max_steps 1
    #--predict_with_generate \
