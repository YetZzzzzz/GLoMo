import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch.nn import L1Loss, MSELoss
from torch.autograd import Function
from math import pi, log
from functools import wraps
from torch import nn, einsum
import torch.nn.functional as F
import os
import time
import random
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_prepare import MMSAATBaselineDataset
from modules.position_embedding import SinusoidalPositionalEmbedding
from MoE import MoE, MLP
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.activations import gelu, gelu_new
from transformers import BertConfig
import numpy as np
import torch.optim as optim
from transformers.optimization import AdamW
from itertools import chain
from torchmetrics import Accuracy, ConfusionMatrix
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
import argparse

from modules.transformer import TransformerEncoder


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

DEVICE = torch.device("cuda:0")

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
]

max_len = 50 # 80
labels_eng =  ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] # 

def pad_collate(batch):
    (x_t, x_a, x_v, y_t, y_a, y_v, y_m) = zip(*batch) # 
    x_t = torch.stack(x_t, dim=0)
    y_t = torch.tensor(y_t)
    y_a = torch.tensor(y_a)
    y_v = torch.tensor(y_v)
    y_m = torch.tensor(y_m)
    x_v = torch.stack(x_v, dim=0)
    x_a_pad = pad_sequence(x_a, batch_first=True, padding_value=0)
    len_trunc_a = min(x_a_pad.shape[1], max_len)
    x_a_pad = x_a_pad[:, 0:len_trunc_a, :]
    len_com_a = max_len - len_trunc_a
    zeros_a = torch.zeros([x_a_pad.shape[0], len_com_a, x_a_pad.shape[2]], device='cpu')
    x_a_pad = torch.cat([x_a_pad, zeros_a], dim=1)

    return x_t, x_a_pad, x_v, y_t, y_a, y_v, y_m
    

    
class Audio_Video_network(nn.Module): 
    def __init__(self, modality_dim, grans, args = None):
        super(Audio_Video_network, self).__init__()
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.d_l = args.d_l
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_dropout = args.attn_dropout
        self.modality_dim = modality_dim # for 
        self.grans = grans
        self.projs = nn.Conv1d(self.modality_dim, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgmaxpoolings_c = nn.AdaptiveMaxPool1d(1)
        self.avgmaxpoolings_f = nn.AdaptiveMaxPool1d(self.grans)
        
        self.encoder = TransformerEncoder(embed_dim=self.d_l,
                                  num_heads= self.num_heads,
                                  layers=self.layers,
                                  attn_dropout= self.attn_dropout,
                                  relu_dropout=self.relu_dropout,   
                                  res_dropout= self.res_dropout,    
                                  embed_dropout=self.embed_dropout,  
                                  attn_mask= False)                 
    def forward(self,feas):
        # the modality dimension can not be divided by num_heads, so first use Conv1d to change the dimensions
        feas = feas.transpose(1, 2)
        feas = self.projs(feas) # , self.modality_dim
        feas = feas.permute(2, 0, 1)
        outputs = self.encoder(feas)# output: [src_len, batch, modality] # [4, 60, 36, 64]
        outputs = outputs.permute(1,2,0)
        coarsed = self.avgmaxpoolings_c(outputs)
        coarsed = coarsed.view(-1,self.d_l)#.squeeze()
        fine_output = self.avgmaxpoolings_f(outputs)
        return coarsed, fine_output



# Attention implementation     
class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):# layers,0.1
        super(Attention, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.attention_mlp = nn.Sequential()
        self.attention_mlp.add_module('attention_mlp', nn.Linear(in_features=dim*3, out_features=hidden_dim))
        self.attention_mlp.add_module('attention_mlp_dropout', nn.Dropout(dropout))
        self.attention_mlp.add_module('attention_mlp_activation', nn.ReLU())
        self.fc_att = nn.Linear(hidden_dim, 3)

    def forward(self, feas1, feas2, feas3):
        multi_hidden1 = torch.cat([feas1, feas2, feas3], dim=1) # [bsz, 768*2]
        attention = self.attention_mlp(multi_hidden1) # [bsz, 64]  
        attention = self.fc_att(attention)# [bsz, 2]
        attention = torch.unsqueeze(attention, 2) * self.scale # [bsz, 2, 1]
        # print('attention1_shape',attention.shape)
        attention = attention.softmax(dim = 1)
        # print('attention2_shape',attention.shape)
        multi_hidden2 = torch.stack([feas1, feas2, feas3], dim=2) # [bsz, 768, 2]
        fused_feat = torch.matmul(multi_hidden2, attention) # 
        fused_feat = fused_feat.squeeze() # [bsz, 64]
        fused_feat = fused_feat.view(-1,self.dim)
        return attention, fused_feat


class FCMoE_cherma(nn.Module,):
    def __init__(self, args, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width_t_ori = 1024 # 768
        self.width_a_ori = 1024 # 768
        self.width_v_ori = 2048 # 512
        self.fea_len_t = args.fea_len_t # DEFAULT 81
        self.fea_len_a = args.fea_len_a # DEFAULT 51
        self.fea_len_v = args.fea_len_v # DEFAULT 17
        self.fea_len_m = args.fea_len_m # DEFAULT 4
        self.num_labels = args.num_classes # DEFAULT 7
        self.gran_t = args.gran_t
        self.gran_a = args.gran_a
        self.gran_v = args.gran_v
        self.d_l = args.d_l
        self.activation = nn.ReLU()       
        self.attn_dropout = args.attn_dropout   #
        self.num_heads = args.num_heads #
        self.relu_dropout = args.relu_dropout #
        self.res_dropout = args.res_dropout #
        self.embed_dropout = args.embed_dropout #
        self.text_network = Audio_Video_network(self.width_t_ori, args.gran_t, args)
        self.audio_network = Audio_Video_network(self.width_a_ori, args.gran_a, args)
        self.video_network = Audio_Video_network(self.width_v_ori, args.gran_v, args)
        self.moe_t = MoE(input_size=args.d_l*self.gran_t, output_size=args.d_l, num_experts=args.experts_t, hidden_size=args.d_l, model=MLP, k=args.k)
        self.moe_a = MoE(input_size=args.d_l*self.gran_a, output_size=args.d_l, num_experts=args.experts_a, hidden_size=args.d_l, model=MLP, k=args.k)
        self.moe_v = MoE(input_size=args.d_l*self.gran_v, output_size=args.d_l, num_experts=args.experts_v, hidden_size=args.d_l, model=MLP, k=args.k) 
         ## multimodal fusion
        self.fc_ts = nn.Linear(in_features=self.d_l*2, out_features=self.d_l)
        self.fc_as = nn.Linear(in_features=self.d_l*2, out_features=self.d_l)
        self.fc_vs = nn.Linear(in_features=self.d_l*2, out_features=self.d_l)
        self.fc_all = nn.Linear(in_features=self.d_l*2, out_features=self.d_l)
        self.attn_cs = Attention(self.d_l, self.d_l)
        self.attn_fs = Attention(self.d_l, self.d_l)
        self.classifier = nn.Linear(in_features=self.d_l*2, out_features= self.num_labels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_l, nhead=self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # num_layers 

    def forward(
        self,
        x_t,
        x_a,
        x_v,
        label_t,
        label_a,
        label_v,
        label_m
    ):
        label_t = label_m
        label_a = label_m
        label_v = label_m
        
        x_t = x_t[:, 0:80, :]
        x_v = x_v.to(torch.float32)
        x_t = x_t.to(torch.float32)
        x_a = x_a.to(torch.float32)

        coarsed_a, finegrained_a = self.audio_network(x_a)
        coarsed_v, finegrained_v = self.video_network(x_v)
        coarsed_t, finegrained_t = self.text_network(x_t) 
        fine_ts, moe_loss_ts = self.moe_t(torch.flatten(finegrained_t, start_dim=1))
        fine_as, moe_loss_as = self.moe_a(torch.flatten(finegrained_a, start_dim=1))
        fine_vs, moe_loss_vs = self.moe_v(torch.flatten(finegrained_v, start_dim=1))
        all_feas = torch.stack((coarsed_t, fine_ts, coarsed_a, fine_as, coarsed_v, fine_vs), dim=0)
        h = self.transformer_encoder(all_feas)
        attn_weights_cs, attn_cs = self.attn_cs(h[0], h[2], h[4])
        attn_weights_fs, attn_fs = self.attn_fs(h[1], h[3], h[5])
        fea_cfs = self.fc_all(torch.cat((attn_cs,attn_fs),dim=1))
        
        fea_modality_t = self.fc_ts(torch.cat((h[0], h[1]), dim=1))
        fea_modality_a = self.fc_as(torch.cat((h[2], h[3]), dim=1))
        fea_modality_v = self.fc_vs(torch.cat((h[4], h[5]), dim=1))
        
        feas_cf_all = torch.stack([fea_modality_t, fea_modality_a, fea_modality_v], dim=2) # [bsz, 768, 2]
        feas_cf_all = torch.matmul(feas_cf_all, attn_weights_cs) 
        feas_cf_all = feas_cf_all.squeeze() # [bsz, 64]
        feas_cf_all = feas_cf_all.view(-1,fea_modality_t.shape[1])
        
        outputs = torch.cat((fea_cfs, feas_cf_all), dim=1)
        logits = self.classifier(outputs)
        moes = moe_loss_ts + moe_loss_as + moe_loss_vs
        all_losses = moes
        return all_losses, logits
        
    def test(self,
            x_t,
            x_a,
            x_v):
        
        x_t = x_t[:, 0:80, :]
        x_v = x_v.to(torch.float32) # [24, 16, 2048] 
        x_t = x_t.to(torch.float32) # [24, 80, 1024] here 24 denotes the batch_size
        x_a = x_a.to(torch.float32) # [24, 80, 1024] 
        
        coarsed_a, finegrained_a = self.audio_network(x_a)
        coarsed_v, finegrained_v = self.video_network(x_v)
        coarsed_t, finegrained_t = self.text_network(x_t) 
        fine_ts, moe_loss_ts = self.moe_t(torch.flatten(finegrained_t, start_dim=1))
        fine_as, moe_loss_as = self.moe_a(torch.flatten(finegrained_a, start_dim=1))
        fine_vs, moe_loss_vs = self.moe_v(torch.flatten(finegrained_v, start_dim=1))

        all_feas = torch.stack((coarsed_t, fine_ts, coarsed_a, fine_as, coarsed_v, fine_vs), dim=0)
        h = self.transformer_encoder(all_feas)
        
        attn_weights_cs, attn_cs = self.attn_cs(h[0], h[2], h[4])
        attn_weights_fs, attn_fs = self.attn_fs(h[1], h[3], h[5])
        fea_cfs = self.fc_all(torch.cat((attn_cs,attn_fs),dim=1))
        
        fea_modality_t = self.fc_ts(torch.cat((h[0], h[1]), dim=1))
        fea_modality_a = self.fc_as(torch.cat((h[2], h[3]), dim=1))
        fea_modality_v = self.fc_vs(torch.cat((h[4], h[5]), dim=1))
        feas_cf_all = torch.stack([fea_modality_t, fea_modality_a, fea_modality_v], dim=2) # [bsz, 768, 2]
        feas_cf_all = torch.matmul(feas_cf_all, attn_weights_cs) 
        
        feas_cf_all = feas_cf_all.squeeze() # [bsz, 64]
        feas_cf_all = feas_cf_all.view(-1,fea_modality_t.shape[1])
        outputs = torch.cat((fea_cfs, feas_cf_all), dim=1)
        logits = self.classifier(outputs)
        moes = moe_loss_ts + moe_loss_as + moe_loss_vs
        all_losses = moes
        return all_losses, logits


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
 
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class Trainer():
    def __init__(self, args):
        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.log_interval = args.log_interval
        # self.local_rank  = args.local_rank
        num_classes = args.num_classes
        self.num_classes = args.num_classes
 
        self.model = FCMoE_cherma(args)#
        self.model = self.model.to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        
        train_data = MMSAATBaselineDataset('train')
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size, collate_fn=pad_collate)
        test_data = MMSAATBaselineDataset('test')
        testdata_sampler = SequentialSampler(test_data)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, collate_fn=pad_collate)
        self.train_te_dataloader = DataLoader(train_data, batch_size=self.batch_size, collate_fn=pad_collate)
        
        self.test_accuracy = Accuracy(task='multiclass',num_classes=num_classes)
        self.test_confmat = ConfusionMatrix(task="multiclass",num_classes=num_classes)

        self.test_pred = []
        self.test_label = []
            
    def train(self):
        self.model.train()
        loss_test_m = []
        acc_test_m = []
        test_loss, test_acc, _ = self.test(self.test_dataloader)# modify the self.test
        self.model.train()
        loss_test_m.append(test_loss)
        acc_test_m.append(test_acc)

        for epoch in range(0, self.epoch):
            for batch_idx, batch in enumerate(self.train_dataloader):
               # self.optimizer.zero_grad()
                text, audio, video, label_t, label_a, label_v, label_m = batch
                
                label_m = label_m.to(device)
                label_m_onehot = F.one_hot(label_m, self.num_classes)

                text = text.to(device)
                audio = audio.to(device)
                video = video.to(device)

                losses, logits = self.model(text, audio, video, label_t, label_a, label_v, label_m)#       
                loss_m = F.cross_entropy(logits, label_m) # 
                
                loss = loss_m + losses
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        epoch, batch_idx * self.batch_size, len(self.train_dataloader.dataset),
                            100. * batch_idx / len(self.train_dataloader)))
                    print('\n Train set: loss_m: {:.4f}\n'.format(loss.item()))

            test_loss, test_acc, p = self.test(self.test_dataloader)#
            # save_name = './glomo_mer/output/Conep' + str(epoch) + '.npy'
            # np.save(save_name, p)# 
            # save_name='glomo'
            # torch.save(self.model.state_dict(), f'./glomo_mer/saved_models/{save_name}_{str(epoch)}_Con.pth')
            self.model.train()

            loss_test_m.append(loss)
            acc_test_m.append(logits)


        loss_test = [loss_test_m]
        acc_test = [acc_test_m]
        return loss_test, acc_test
    
    def test(self, dataloader):
        self.model.eval()
       
        loss_m = 0
        test_loss = 0
        cor_m = 0
        predicted = []
        all_label_m = []
        p = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                text, audio, video, label_t, label_a, label_v, label_m = batch
                label_t, label_a, label_v, label_m = label_t.to(device), label_a.to(device), label_v.to(device), label_m.to(device)

                text = text.to(device)
                audio = audio.to(device)
                video = video.to(device)

                losses, logits = self.model.test(text, audio, video)
                if batch_idx == 0:
                    p = np.array(logits.cpu().numpy())
                else:
                    p = np.concatenate((p, logits.cpu().numpy()), axis=0)
                       
                loss_m += F.cross_entropy(logits, label_m, reduction ='sum').item()

                pred = logits.argmax(dim=1, keepdim=True)  
                cor_m += pred.eq(label_m.view_as(pred)).sum().item()

                predicted.extend(logits.cpu().numpy().argmax(1))
                all_label_m.extend(label_m.cpu().numpy())
                
            self.test_pred.extend(pred.tolist())
            self.test_label.extend(label_m.tolist())


        print('accuracy: ', self.test_accuracy)
        print('confusion matrix: ', self.test_confmat)

        c_m = confusion_matrix(all_label_m, predicted)#,  normalize='true')
        c_m_n = confusion_matrix(all_label_m, predicted,  normalize='true')
        c_r = classification_report(all_label_m, predicted, target_names = labels_eng, digits = 4)

        print(c_m)
        print(c_r)

        disp = ConfusionMatrixDisplay(confusion_matrix=c_m_n, display_labels = labels_eng)

        test_len = len(dataloader.dataset)
        cor_m /= test_len
        loss_m /= test_len

        print('\nTest set: loss_m: {:.4f},  Acc_m: {:.4f} \n'.format(loss_m, cor_m))
        
        return loss_m, cor_m, p



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GLoMo')
    parser.add_argument("--gran_t", type=int, default=3)
    parser.add_argument("--gran_a", type=int, default=3)
    parser.add_argument("--gran_v", type=int, default=3)
    parser.add_argument("--TEXT_DIM", type=int, default=768)
    parser.add_argument("--ACOUSTIC_DIM", type=int, default=768)
    parser.add_argument("--VISUAL_DIM", type=int, default=512)
    parser.add_argument("--experts_t", type=int, default=3)
    parser.add_argument("--experts_a", type=int, default=3)
    parser.add_argument("--experts_v", type=int, default=3)
    parser.add_argument("--experts_all", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--k_all", type=int, default=2)
    parser.add_argument("--attn_dropout", type=float, default=0.5)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--relu_dropout", type=float, default=0.3)
    parser.add_argument("--res_dropout", type=float, default=0.3)
    parser.add_argument("--embed_dropout", type=float, default=0.2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--model_path", type=str, default='GLOMO_cherma.pth')
    parser.add_argument('--max_len', default=50, type=int, help='maximum length for audio sequence')# 80
    parser.add_argument('--num_classes', default=7, type=int, help='number of emotions')
    parser.add_argument('--fea_len_t', default=81, type=int, help='dimension of the feature vector of text')
    parser.add_argument('--fea_len_a', default=51, type=int, help='dimension of the feature vector of audio')
    parser.add_argument('--fea_len_v', default=17, type=int, help='dimension of the feature vector of visual')
    parser.add_argument('--fea_len_m', default=4, type=int, help='dimension of the feature vector of multi-modality')
    parser.add_argument('--epoch', default=50, type=int, help='number of training epoches')
    parser.add_argument('--batch_size', default=400, type=int, help='batch_size for training')#
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument("--learning_rate", type=float, default=2e-5)# 2e-5
    parser.add_argument("--seed", type=int, default=5576)
    parser.add_argument("--d_l", type=int, default=1024)# 128
    parser.add_argument("--dropout_prob", type=float, default=0.5) # 0.5
     
    args = parser.parse_args()
    args.labels_eng = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    set_random_seed(args.seed)

    device = torch.device('cuda')


    tic =time.time()
    a = Trainer(args)# 
    print(a)
    loss_test, acc_test = a.train()

    toc = time.time()
    runtime = toc - tic
    print('running time: ', runtime)
        



  
