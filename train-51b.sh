export PJRT_DEVICE=TPU
# Default is 0, set to 1 for debugging
export PT_XLA_DEBUG=0
export USE_TORCH=ON
export TPU_NUM_DEVICES=4


export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1


python3 -u examples/pytorch/xla_spawn.py    \
--num_cores 4 examples/pytorch/language-modeling/run_clm.py     \
--num_train_epochs 500   \
--dataset_name wikitext      \
--dataset_config_name wikitext-2-raw-v1      \
--per_device_train_batch_size 4     \
--per_device_eval_batch_size 4  \
--do_train   \
--tensor_parallel_size 8 \
--use_fsdp  \
--use_nested_fsdp    \
--output_dir /tmp/test-clm      \
--overwrite_output_dir  \
--config_name /home/sivaibhav/transformers/config-51b.json \
--cache_dir /tmp  \
--tokenizer_name gpt2   \
--block_size 2048 \
--optim adafactor  \
--adafactor true   \
--use_grad_ckpt  \
--save_strategy no  \
--logging_strategy no &> train.log

