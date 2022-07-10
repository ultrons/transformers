export USE_TORCH=ON
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

python examples/pytorch/xla_spawn.py \
  --num_cores 8 \
  examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-3b \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
