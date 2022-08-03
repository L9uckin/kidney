# model settings
_base_ = [
    '../_base_/datasets/feng2022.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=4,
        strides=(1, 1, 1,  1),
        enc_num_convs=(2,  2, 2, 2),
        dec_num_convs=(2, 2, 2),
        downsamples=(True, True, True),
        enc_dilations=(1, 1,  1, 1),
        dec_dilations=(1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    neck=dict(
        type='FPN',
        in_channels=[512, 256 ,128 ,64],
        out_channels=64,
        num_outs=4),
    decode_head=dict(
            type='FPNHead',
            in_channels=[64, 64, 64, 64],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=64,
            dropout_ratio=-1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

runner = dict(type='IterBasedRunner', max_iters=160000)
data = dict(
        samples_per_gpu=2,
        workers_per_gpu=4,)
evaluation = dict(interval=1600,metric= ['mIoU', 'mDice', 'mFscore'])