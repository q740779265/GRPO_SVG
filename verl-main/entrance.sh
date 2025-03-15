set -x
export NCCL_P2P_DISABLE=1
export FLASH_ATTENTION_SKIP_CUDA_BUILD=1

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7
# export VLLM_TORCH_COMPILE_LEVEL=0

python3 -m verl.trainer.main_ppo \
    config_path=./verl/trainer/config/grpo_trainer.yaml \
    data.train_files=/home/wxf/lzq/twq/verl-main/examples/datasets/svg/train.parquet \
    data.val_files=/home/wxf/lzq/twq/verl-main/examples/datasets/svg/train.parquet\
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/home/wxf/lzq/data/model/qwen/qwen2.5-coder-7b \
    reward_model.model.path=/home/wxf/lzq/data/model/paligemma2-10b-mix-448/models--google--paligemma2-10b-mix-448/snapshots/b26d16fb4251090ba4a4aa5af9fca1f8248ed5b6\
    trainer.project_name='verl_grpo_SVG' \
    trainer.experiment_name='qwen25_05' \
    trainer.n_gpus_per_node=4 \

# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=/home/ubuntu/Documents/twq/verl/verl-main/examples/datasets/bespoke/train.parquet \
#     data.val_files=/home/ubuntu/Documents/twq/verl/verl-main/examples/datasets/bespoke/test.parquet \
#     data.train_batch_size=4 \
#     data.val_batch_size=128 \
#     data.max_prompt_length=512 \
#     data.max_response_length=1024 \
#     actor_rollout_ref.model.path=/home/ubuntu/Documents/wulei/model/Qwen/Qwen2.5-3B-Base \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=4 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.grad_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=3 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=6 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console'] \
#     trainer.project_name='verl_grpo_bespoke' \
#     trainer.experiment_name='qwen2.5_3b_function_rm' \
#     trainer.n_gpus_per_node=1 \
#     trainer.nnodes=3 \
#     trainer.save_freq=-1 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=15 