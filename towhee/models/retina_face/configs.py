model_cfgs = {
    'cfg_mnet' : {
        'name': 'mobilenet0.25',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 32,
        'ngpu': 1,
        'epoch': 250,
        'decay1': 190,
        'decay2': 220,
        'image_size': 640,
        'target_size': 1600,
        'nms_threshold': 0.4,
        'confidence_threshold':0.5,
        'max_size': 2150,
        'mean': [104, 117, 123],
        'std': [1,1,1],
        'pretrain': True,
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64
    },

    'cfg_re50' : {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'target_size': 1600,
        'nms_threshold': 0.4,
        'confidence_threshold':0.5,
        'max_size': 2150,
        'mean': [104, 117, 123],
        'std': [1,1,1],
        'pretrain': True,
        'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
        'in_channel': 256,
        'out_channel': 256
    }
}

def build_configs(name: str):
    return model_cfgs[name]
