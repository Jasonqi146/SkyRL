set -x

rm -r $HOME/.cache
rm -r /tmp/ray

NUM_GPUS=4
NUM_INFERENCE_ENGINES=4
TP_SIZE=1
LOGGER=wandb
DATE=$(date +%Y%m%d_%H%M%S)

# Colocated GRPO training+generation for Qwen2.5-Coder-1.5B-Instruct on GSM8k dataset.
# Uses 1 node with 4 GPUs.
# uv run examples/llm_as_a_judge/gsm8k_dataset_judge.py --output_dir $HOME/data/gsm8k_llm_judge
# add OPENAI_API_KEY and WANDB_API_KEY to .env.llm_judge
# bash examples/llm_as_a_judge/run_llm_judge.sh
export HF_TOKEN=xx
export WANDB_API_KEY=xx
export DATA_DIR="$HOME/SkyRL/skyrl-train/examples/llm_as_a_judge_t_bench"
export CKPT_PATH="$HOME/research_nfs/jasonqi_weights/llm_as_a_judge_t_bench/$DATE"
export HF_HOME="/mnt/huggingface"

# We use a smaller batch size here for demonstration
uv run --isolated --extra vllm --env-file .env.llm_judge -m examples.llm_as_a_judge_t_bench.main_llm_judge \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.epochs=50 \
  trainer.eval_batch_size=4 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=100 \
  trainer.max_prompt_length=20000 \
  generator.sampling_params.max_generate_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="llm_as_a_judge_t_bench" \
  trainer.run_name="llm_as_a_judge_t_bench" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  environment.env_class=llm_as_a_judge_t_bench \
  environment.skyrl_gym.llm_as_a_judge_t_bench.model="o4-mini" \
  $@