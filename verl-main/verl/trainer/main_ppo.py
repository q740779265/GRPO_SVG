# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score.bespoke import compute_score
import ray
import hydra
import subprocess
import os
from omegaconf import OmegaConf

from verl.single_controller.ray import RayWorkerGroup
# 记得添加verl到系统搜索路径
# import sys
# new_path = '/home/wxf/lzq/twq/verl-main'
# sys.path.append(new_path)
def get_gpu_ids():
    cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
    gpu_memory = subprocess.check_output(cmd.split()).decode().strip().split('\n')
    free_gpus = [i for i, mem in enumerate(gpu_memory) if int(mem) < 100]
    if len(free_gpus) == 0:
        raise ValueError("没有可用的GPU")
    print(f"可用的GPU: {free_gpus}")
    os.environ["CUDA_VISIBLE_D0EVICES"] = ",".join(map(str, free_gpus))
    return free_gpus

def main():
    # get_gpu_ids()
    # config_path = 'config_path=./verl/trainer/config/grpo_trainer.yaml'
    # config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_config.config_path)
    config = OmegaConf.merge(file_config, cli_config)
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))

@ray.remote(num_cpus=1, num_gpus=2)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_local_path_from_hdfs
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    global_pool_id_1 = 'global_pool_1'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        # global_pool_id_1: [2],
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    from svg_custom.custom_reward_model import RewardModelWorker
    

    if config.reward_model.enable:
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # reward_manager_name = config.reward_model.get("reward_manager", "naive")
    # if reward_manager_name == 'naive':
    #     from verl.workers.reward_manager import NaiveRewardManager
    #     reward_manager_cls = NaiveRewardManager
    # elif reward_manager_name == 'prime':
    #     from verl.workers.reward_manager import PrimeRewardManager
    #     reward_manager_cls = PrimeRewardManager
    # else:
    #     raise NotImplementedError
    from svg_custom.custom import custom_manager
    # from verl.workers.reward_manager.naive import NaiveRewardManager
    print("reward_fn初始化")
    reward_fn = custom_manager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    # val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=RayWorkerGroup,
                            reward_fn=reward_fn,
                            val_reward_fn=reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
