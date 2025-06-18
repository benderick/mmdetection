_base_ = './dino-4scale_r50_8xb2-12e_coco.py'
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)


num_levels = 3
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='mmpretrain.MViT',
        arch='tiny',
        out_scales=[1, 2, 3],
        drop_path_rate=0.1),
    neck=dict(in_channels=[192, 384, 768], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))

train_dataloader = dict(
    batch_size=4)
val_dataloader = dict(
    batch_size=4)
test_dataloader = dict(
    batch_size=4)