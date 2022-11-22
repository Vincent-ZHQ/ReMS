import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
# from models.text.BertTextEncoder import BertTextEncoder

__all__ = ['REMS']
class REMS(nn.Module):
    def __init__(self, args):
        super(REMS, self).__init__()
        # text subnets
        self.args = args
        self.nModality = max(len(args.tasks) - 1, 1)
        self.weight_k = args.weight_k
        if self.args.bert_type in ['bert_base']:
            self.text_model = BertModel.from_pretrained('/scratch/users/ntu/n2107335/Multimodal/Pretrained/bert_base_uncased/')
            # self.text_model = BertModel.from_pretrained("bert_base_uncased")
        else:
            # self.text_model = BertModel.from_pretrained("bert_large_uncased")
            self.text_model = BertModel.from_pretrained(
                '/scratch/users/ntu/n2107335/Multimodal/Pretrained/bert_large_uncased/')
        self.audio_model = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, dropout=args.post_audio_dropout, batch_first=True,
                                   bidirectional=True)  # bidirectional = True

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_dim) #args.post_text_dim
        self.post_text_layer_2 = nn.Linear(args.post_dim, args.post_dim)
        self.post_text_layer_3 = nn.Linear(args.post_dim, args.output_dim)
        
        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)#p=args.post_audio_dropout
        self.audio_mp = nn.MaxPool2d((3, 1), stride=(2, 1))
        self.post_audio_layer_1 = nn.Linear(102400, args.post_dim)#args.post_audio_dim #205312
        self.post_audio_layer_2 = nn.Linear(args.post_dim, args.post_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_dim, args.output_dim)
        
        # the classify layer for video
        self.video_mp = nn.MaxPool2d((3, 1), stride=(2, 1))
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(11776, args.post_dim)
        self.post_video_layer_2 = nn.Linear(args.post_dim, args.post_dim)
        self.post_video_layer_3 = nn.Linear(args.post_dim, args.output_dim)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(self.nModality*args.post_dim, args.post_fusion_dim) #
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, args.output_dim)

        # the weight generate
        self.post_weight_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_weight_layer_1 = nn.Linear(self.nModality*args.post_dim, args.post_fusion_dim)
        self.post_weight_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_weight_layer_3 = nn.Linear(args.post_fusion_dim, 3)


    def forward(self, text=None, audio=None, video=None, label=None):

        res = {}
        text_h, audio_h, video_h, text_f, audio_f, video_f = None, None, None, None, None, None

        if 'V' in self.args.tasks:
            video = self.video_mp(video)
            video = torch.flatten(video, 1)

            # vision output
            video_h = self.post_video_dropout(video)
            video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)
            x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
            output_video = self.post_video_layer_3(x_v)

            video_f = x_v

            res.update({
                'M': output_video,
                'V': output_video,
            })

        if 'A' in self.args.tasks:
            # audio
            audio = torch.squeeze(audio, 1)
            audio, _ = self.audio_model(audio)  # [batch, 300, 512]
            audio = self.audio_mp(audio)
            audio = torch.flatten(audio, 1)

            # audio output
            audio_h = self.post_audio_dropout(audio)
            audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
            x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
            output_audio = self.post_audio_layer_3(x_a)
            audio_f = x_a

            res.update({
                'M': output_audio,
                'A': output_audio
            })

        if 'T' in self.args.tasks:
            input_ids = torch.squeeze(text[:, 0, :], 1)
            input_mask = torch.squeeze(text[:, 1, :], 1)
            segment_ids = torch.squeeze(text[:, 2, :], 1)

            text = self.text_model(input_ids=input_ids.long(), attention_mask=input_mask.long(), token_type_ids=segment_ids.long())[0]

            ## text
            text_d = self.post_text_dropout(text[:, 0, :])
            text_h = F.relu(self.post_text_layer_1(text_d), inplace=False) # (32, 128)
            x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
            output_text = self.post_text_layer_3(x_t)

            text_f = x_t

            res.update({
                'M': output_text,
                'T': output_text
            })

        if self.args.rems_use:
            features_h = {'T': text_h, 'A': audio_h, 'V': video_h}
            fusion_h = [features_h[i] for i in self.args.tasks[1:]]
            fusion_w = torch.cat(fusion_h, dim=-1)

            # fusion output
            fusion_w = self.post_weight_dropout(fusion_w)
            fusion_w = F.relu(self.post_weight_layer_1(fusion_w), inplace=False)

            # classifier-fusion
            x_w = F.relu(self.post_weight_layer_2(fusion_w), inplace=False)
            output_w = self.post_weight_layer_3(x_w)

            # weight fusion
            if label is not None:
                weight_text, weight_audio, weight_video = weight_fun(self.args.dataset, output_text, output_audio, output_video, label, self.weight_k)
            else:
                weight_text, weight_audio, weight_video = output_w[:, 0].view(-1,1), output_w[:, 1].view(-1,1), output_w[:, 2].view(-1,1)

            weight_text = weight_text.to(self.args.device)
            weight_audio = weight_audio.to(self.args.device)
            weight_video = weight_video.to(self.args.device)

            text_f = weight_text.to(torch.float32) * x_t # text_h
            audio_f = weight_audio.to(torch.float32) * x_a # audio_h
            video_f = weight_video.to(torch.float32) * x_v # video_h

            weight = torch.cat([weight_text, weight_audio, weight_video], dim=1)

            res.update({
                'OP_W': output_w,
                'GT_W': weight
            })

        if 'M' in self.args.tasks:
            features_f = {'T': text_f, 'A': audio_f, 'V': video_f}
            fusion_m = torch.cat([features_f[i] for i in self.args.tasks[1:]], dim=-1)  # use

            # fusion output
            fusion_m = self.post_fusion_dropout(fusion_m)
            fusion_m = F.relu(self.post_fusion_layer_1(fusion_m), inplace=False)

            # classifier-fusion
            x_m = F.relu(self.post_fusion_layer_2(fusion_m), inplace=False)
            output_multimodal = self.post_fusion_layer_3(x_m)

            res.update({
                'M': output_multimodal,
            })
        return res


def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)

def dis_entropy(y_pre,y):
    y = y.reshape(-1)
    exp_pre = softmax(y_pre)
    return exp_pre[range(y.shape[0]),y]

def weight_fun(dataset, output_text, output_audio, output_video, ground_truth, weight_k=2):
    
    ground_truth = ground_truth.view(-1,1)
    ground_truth = ground_truth.cpu().detach().numpy()  

    output_audio = output_audio.cpu().detach().numpy()
    output_video = output_video.cpu().detach().numpy()
    output_text = output_text.cpu().detach().numpy()
    
    if ground_truth.shape[0] != output_audio.shape[0]:
        ground_truth = ground_truth[:output_audio.shape[0]]

    if dataset in ['iemocap']:
        dis_audio = dis_entropy(output_audio, ground_truth)
        dis_video = dis_entropy(output_video, ground_truth)
        dis_text = dis_entropy(output_text, ground_truth)
    else:
        dis_audio = abs(output_audio - ground_truth)
        dis_video = abs(output_video - ground_truth)
        dis_text = abs(output_text - ground_truth)

    exp_audio = np.exp(-weight_k*dis_audio)
    exp_video = np.exp(-weight_k*dis_video)
    exp_text = np.exp(-weight_k*dis_text)
    
    exp_sum = exp_audio + exp_video + exp_text

    weight_text = exp_text / exp_sum   
    weight_audio = exp_audio / exp_sum
    weight_video = exp_video / exp_sum

    return torch.from_numpy(weight_text).view(-1,1), torch.from_numpy(weight_audio).view(-1,1), torch.from_numpy(weight_video).view(-1,1)