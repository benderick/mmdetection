_base_ = 'mmdet::common/rice.py'

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

# model settings
model = dict(
    type='BoxInst',
    data_preprocessor=dict(
        type='BoxInstDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        mask_stride=4,
        pairwise_size=3,
        pairwise_dilation=2,
        pairwise_color_thresh=0.3,
        bottom_pixels_removed=10),
    backbone=dict(
        type='mmpretrain.SwinTransformerV2_MBFD',
        arch='tiny',
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='BoxInstBboxHead',
        num_params=593,
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    mask_head=dict(
        type='BoxInstMaskHead',
        num_layers=3,
        feat_channels=16,
        size_of_interest=8,
        mask_out_stride=4,
        topk_masks_per_img=64,
        mask_feature_head=dict(
            in_channels=256,
            feat_channels=128,
            start_level=0,
            end_level=2,
            out_channels=16,
            mask_stride=8,
            num_stacked_convs=4,
            norm_cfg=dict(type='BN', requires_grad=True)),
        loss_mask=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            eps=5e-6,
            loss_weight=1.0)),
    # model training and testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr=0.5))

# optimizer
# optim_wrapper = dict(optimizer=dict(lr=0.01))

# # evaluator
# val_evaluator = dict(metric=['bbox', 'segm'])
# test_evaluator = val_evaluator
