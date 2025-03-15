## 并行相关参数说明
```yaml
actor:
    # 加速策略，可选fsdp（默认）和megatron
    strategy: fsdp  
    # 使用动态批量大小，每个batch的大小根据这个batch的总token数分配
    use_dynamic_bsz: False
    # 序列并行大小，将sequence拆分到多个gpu上处理，每个gpu加载整个模型
    ulysses_sequence_parallel_size: 2 # sp size
    fsdp_config:
        # offload：将参数从gpu上卸载到内存里，需要的时候再读
        # 参数卸载
        param_offload: True
        # 梯度卸载
        grad_offload: True
        # 优化器卸载
        optimizer_offload: True
        # fsdp大小，只能等于-1或＞world_size，不然报错
        fsdp_size: -1
rollout:
    # 从gpu释放vllm引擎的缓存
    free_cache_engine: True
    # load_format: The format of the model weights to load:
    #     "auto" will try to load the weights in the safetensors format and
    #         fall back to the pytorch bin format if safetensors format is
    #         not available.
    #     "pt" will load the weights in the pytorch bin format.
    #     "safetensors" will load the weights in the safetensors format.
    #     "npcache" will load the weights in pytorch format and store
    #         a numpy cache to speed up the loading.
    #     "dummy" will initialize the weights with random values, which is
    #         mainly for profiling.
    #     "tensorizer" will use CoreWeave's tensorizer library for
    #         fast weight loading.
    #     "bitsandbytes" will load nf4 type weights.
    # 加载格式
    load_format: dummy_dtensor
    # 张量并行，将模型权重分到多个设备上进行训练，每个设备计算完整序列的部分特征
    tensor_model_parallel_size: 2
```