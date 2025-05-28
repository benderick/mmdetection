_base_ = './dino-4scale_r50_8xb2-12e_coco.py'
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

image_size = (640, 640)

num_levels = 5
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt_MBFD',
        arch='tiny',
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        layer_scale_init_value=0.,
        use_grn=True,
        gap_before_final_norm=False,
        init_cfg=None),
    neck=dict(in_channels=[96, 192, 384, 768], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))

train_dataloader = dict(
    batch_size=3)
val_dataloader = dict(
    batch_size=3)
test_cfg_dataloader = dict(
    batch_size=3)
