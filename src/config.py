import os
import argparse

from src.utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'rems': self.__REMS
        }
        # normalize
        model_name = str.lower(args.modelname)
        dataset_name = str.lower(args.dataset)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
 
    def __REMS(self):
        tmp = {
            'commonParas':{
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
            },
            # dataset
            'datasetParas':{
                'iemocap': {
                    # training/validation/test parameters
                    'early_stop': 4,
                    'batch_size': 32,

                    'lr_audio': 5e-6,
                    'lr_video': 5e-4,
                    # 'learning_rate_bert': 2e-5,
                    # 'learning_rate_other': 5e-5,
                    # 'lr_other': 
                    'weight_decay': 0.001,

                    # dim
                    'text_out': 768, # 1024 for bert-large; 768 for bert-base
                    'audio_out': 400,
                    'video_out': 400,
                    'post_dim': 128,
                    'post_fusion_dim': 128,
                    'output_dim': 4,

                    # 'post_fusion_dropout': 0.1,
                    # 'post_text_dropout': 0.0,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.0,
                    #
                    'weight_k': 2,
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args