# 修改数据集相关配置
data_type = 'CocoDataset'
data_root = '/icislab/volume3/benderick/futurama/openmmlab/mmdetection/data/VisDrone-bak/'
metainfo = {
    'classes': ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'),
}

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',)
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=data_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-train/annotations/train.json',
        data_prefix=dict(img='VisDrone2019-DET-train/images/'),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=data_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-val/annotations/val.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=data_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-test-dev/annotations/test.json',
        data_prefix=dict(img='VisDrone2019-DET-test-dev/images/'),
        pipeline=test_pipeline,
        backend_args=backend_args))

# 修改评价指标相关配置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'VisDrone2019-DET-val/annotations/val.json',
    metric='bbox')
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'VisDrone2019-DET-test-dev/annotations/test.json',
    metric='bbox',)