default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl',)
)

# 模型包装器配置：用于处理分布式训练中的未使用参数问题
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',               # 分布式数据并行包装器
    find_unused_parameters=True,                    # 查找未使用参数：解决DDP训练中某些参数未接收梯度的问题
    broadcast_buffers=False)                        # 不广播缓冲区：提高训练效率，避免不必要的通信开销


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=42, deterministic=False)
