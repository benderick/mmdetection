_base_ = [
    'mmdet::_base_/datasets/rice_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.SparseInst.sparseinst', 'mmpretrain.models'], allow_failed_imports=False)

model = dict(
    type='SparseInst',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmpretrain.SwinTransformerV2',
        arch='tiny',
        drop_path_rate=0.2,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        init_cfg=None),
    encoder=dict(
        type='InstanceContextEncoder',
        in_channels=[192, 384, 768],
        out_channels=256),
    decoder=dict(
        type='BaseIAMDecoder',
        in_channels=256 + 2,
        num_classes=1,
        ins_dim=256,
        ins_conv=4,
        mask_dim=256,
        mask_conv=4,
        kernel_dim=128,
        scale_factor=2.0,
        output_iam=False,
        num_masks=100),
    criterion=dict(
        type='SparseInstCriterion',
        num_classes=1,
        assigner=dict(type='SparseInstMatcher', alpha=0.8, beta=0.2),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            alpha=0.25,
            gamma=2.0,
            reduction='sum',
            loss_weight=2.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            reduction='sum',
            eps=5e-5,
            loss_weight=2.0),
    ),
    test_cfg=dict(score_thr=0.005, mask_thr_binary=0.45))

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    sampler=dict(type='InfiniteSampler'))

max_iters = 20_000
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.00005, weight_decay=0.05))
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=200)
# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        milestones=[max_iters//4, max_iters//2],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=200, max_keep_ckpts=3))
log_processor = dict(by_epoch=False)

auto_scale_lr = dict(base_batch_size=16, enable=True)
