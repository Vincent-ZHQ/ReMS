"""
AIO -- All Trains in One
"""
import logging
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from src.utils.functions import dict_to_str
from src.utils.metricsTop import MetricsTop

__all__ = ['REMSTrain']

logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger('MER')

class REMSTrain():
    def __init__(self, args):
        self.args = args
        self.metrics = MetricsTop().getMetics(args.dataset)
        self.train_epoch = 0
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.entropy = nn.CrossEntropyLoss(weight=torch.tensor([4290 / 1103, 4290 / 1084, 4290 / 1636, 4290 / 1708]).cuda())

    def do_train(self, model, dataloader):

        tfm_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        text_bert_params = list(model.text_model.named_parameters())
        text_bert_params_decay = [p for n, p in text_bert_params if not any(nd in n for nd in tfm_no_decay)]
        text_bert_params_no_decay = [p for n, p in text_bert_params if any(nd in n for nd in tfm_no_decay)]
        text_other_params = [p for n, p in list(model.named_parameters()) if ('post_text' in n)]
        audio_params = [p for n, p in list(model.named_parameters()) if ('audio' in n)]
        video_params = [p for n, p in list(model.named_parameters()) if ('video' in n)]
        model_params_other = [p for n, p in list(model.named_parameters()) if
                              ('text' not in n) and ('audio' not in n) and ('video' not in n)]

        optimizer_grouped_parameters = [
            {'params': text_bert_params_decay, 'weight_decay': self.args.weight_decay, 'lr': self.args.lr_text_bert},
            {'params': text_bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.lr_text_bert},
            {'params': text_other_params, 'weight_decay': self.args.weight_decay, 'lr': self.args.lr_text_other},
            {'params': audio_params, 'weight_decay': self.args.weight_decay, 'lr': self.args.lr_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay, 'lr': self.args.lr_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay, 'lr': self.args.lr_other}
        ]

        optimizer = optim.Adam(optimizer_grouped_parameters)
        train_loader, valid_loader, test_loader = dataloader

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else -1e8
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}

            model.train()
            train_loss_1 = 0.0
            weight_loss = 0.0
            train_loss_2 = 0.0
            with tqdm(train_loader) as td:
                for batch_audio, batch_video, batch_text, batch_labels in td:
                    vision = batch_video.to(self.args.device)
                    audio = batch_audio.to(self.args.device)
                    text = batch_text.to(self.args.device)
                    labels = batch_labels.to(self.args.device).view(-1)

                    # print(vision.shape, audio.shape, text.shape, labels.shape)
                    # MOSI: [48, 512], [401, 40], [3, 50]

                    optimizer.zero_grad()
                    # forward
                    outputs = model(text=text, audio=audio, video=vision, label=labels)

                    if not self.args.two_stage:
                        for m in self.args.tasks:
                            y_pred[m].append(outputs[m].cpu())
                            y_true[m].append(labels.cpu())

                    # compute loss
                    loss_1 = 0.0
                    for m in self.args.tasks:
                        loss_1 += self.weighted_loss(self.args.dataset, outputs[m], labels)

                    if self.args.rems_use:
                        loss_w = self.L1(outputs['OP_W'], outputs['GT_W'])
                        loss_1 += loss_w
                    weight_loss += loss_w.item()

                    loss_1.backward()
                    train_loss_1 += loss_1.item()
                    
                    # update parameters
                    optimizer.step()

                    if self.args.two_stage:
                        optimizer.zero_grad()
                        # forward
                        outputs = model(text=text, audio=audio, video=vision)

                        # store results
                        for m in self.args.tasks:
                            y_pred[m].append(outputs[m].cpu())
                            y_true[m].append(labels.cpu())

                        # compute loss
                        loss_2 = 0.0
                        if 'M' in self.args.tasks:
                            loss_2 = self.weighted_loss(self.args.dataset, outputs['M'], labels)

                        loss_2.backward()
                        train_loss_2 += loss_2.item()
                        optimizer.step()
                        
            print("weight_loss: ", weight_loss)

            train_loss = train_loss_2 / len(train_loader) if self.args.two_stage else train_loss_1 / len(train_loader)
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            # for m in self.args.tasks:
            pred, true = torch.cat(y_pred[self.args.unitask]), torch.cat(y_true[self.args.unitask])
            train_results = self.metrics(pred, true)
            logger.info('%s: >> ' %(self.args.unitask) + dict_to_str(train_results))

            if epochs < self.args.MIN_Epoch:
                continue
            
            # validation
            val_results = self.do_test(model, valid_loader, mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            print(isBetter)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), self.args.best_model_save_path)
                model.to(self.args.device)
            # return self.args.MAX_Epoch

            if epochs > self.args.MAX_Epoch:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), self.args.best_model_save_path)
                return self.args.MAX_Epoch
            
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                self.train_epoch = best_epoch
                return self.train_epoch
            

    def do_test(self, model, dataloader, mode="VAL"):

        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}

        eval_loss = 0.0
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_audio, batch_video, batch_text, batch_labels in td:
                    vision = batch_video.to(self.args.device)
                    audio = batch_audio.to(self.args.device)
                    text = batch_text.to(self.args.device)
                    labels = batch_labels.to(self.args.device).view(-1)

                    outputs = model(text=text, audio=audio, video=vision)

                    loss = self.weighted_loss(self.args.dataset, outputs[self.args.unitask], labels)
                    eval_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels.cpu())
                    
        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
    # for m in self.args.tasks:
        pred, true = torch.cat(y_pred[self.args.unitask]), torch.cat(y_true[self.args.unitask])
        eval_results = self.metrics(pred, true)
        logger.info('%s: >> ' %(self.args.unitask) + dict_to_str(eval_results))

        eval_results_m = self.metrics(torch.cat(y_pred[self.args.unitask]), torch.cat(y_true[self.args.unitask]))
        len(torch.cat(y_true[self.args.unitask]))

        eval_results_m['EPOCH'] = self.train_epoch
        eval_results_m['Loss'] = eval_loss
        
        return eval_results_m
    
    def weighted_loss(self, dataset, y_pred, y_true):
        if dataset in ['iemocap']:
            loss = self.entropy(y_pred, y_true)
        else:
            y_pred = y_pred.view(-1)
            y_true = y_true.view(-1)
            loss = self.L1(y_pred, y_true)

        return loss#.sum()

