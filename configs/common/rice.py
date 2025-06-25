_base_ = '../_base_/default_runtime_v2.py'

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/icislab/volume3/benderick/futurama/openmmlab/mmdetection/data/rice/'
metainfo = {
    'classes': ('plot',),
}
backend_args = None

# Align with Detectron2
backend = 'pillow'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend=backend),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend=backend),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True, backend=backend),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
# val_dataloader = dict(
#     batch_size=6,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     pin_memory=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='annotations/instances_val2017.json',
#         data_prefix=dict(img='val2017/'),
#         test_mode=True,
#         pipeline=test_pipeline,
#         backend_args=backend_args))
test_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
val_dataloader = test_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/instances_val2017.json',
#     metric='segm',
#     format_only=False,
#     backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test2017.json',
    metric='segm',
    format_only=False,
    backend_args=backend_args)

val_evaluator = test_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00001, weight_decay=0.0001),
    )

# learning policy
max_epochs = 150
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30, 60, 100],
        gamma=0.1)
]


auto_scale_lr = dict(base_batch_size=16)

default_hooks = dict(checkpoint=dict(by_epoch=True, interval=1))
log_processor = dict(by_epoch=True)
