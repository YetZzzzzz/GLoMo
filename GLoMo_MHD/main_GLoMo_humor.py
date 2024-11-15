from __future__ import absolute_import, division, print_function
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import pickle
import numpy as np
from typing import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, BCEWithLogitsLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from global_configs_class import *
from GLoMo_humor import GLoMo

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["urfunny", "sarcasm"], default="sarcasm")
parser.add_argument("--max_seq_length", type=int, default=50)# can be higher, like 64 for urfunny and 77 for sarcasm

parser.add_argument("--train_batch_size", type=int, default=64)# 16
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5) 
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased"],
    default="bert-base-uncased",
)
parser.add_argument("--gran_t", type=int, default=3)
parser.add_argument("--gran_a", type=int, default=3)
parser.add_argument("--gran_v", type=int, default=3)
parser.add_argument("--TEXT_DIM", type=int, default=768)
parser.add_argument("--ACOUSTIC_DIM", type=int, default=60)
parser.add_argument("--VISUAL_DIM", type=int, default=36)
parser.add_argument("--experts_t", type=int, default=3)
parser.add_argument("--experts_a", type=int, default=3)
parser.add_argument("--experts_v", type=int, default=3)
parser.add_argument("--experts_all", type=int, default=2)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--k_all", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--seed", type=int, default=5576)
parser.add_argument("--d_l", type=int, default=64)# 128
parser.add_argument("--attn_dropout", type=float, default=0.5)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--relu_dropout", type=float, default=0.3)
parser.add_argument("--res_dropout", type=float, default=0.3)
parser.add_argument("--embed_dropout", type=float, default=0.2)  
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--load", type=int, default=0)
parser.add_argument("--test", type=int, default=0) 
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument('--num_labels', type=int, default=2)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay') # 0.01
parser.add_argument('--schedule', default=[80, 100], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')# needs to adjust based on n_epochs []
parser.add_argument("--model_path", type=str, default='glomo.pth')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
args = parser.parse_args()

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected. Recieved {0}".format(s)
        )


def seed(s):
    if isinstance(s, int):
        if 0 <= s <= 9999:
            return s
        else:
            raise argparse.ArgumentTypeError(
                "Seed must be between 0 and 2**32 - 1. Received {0}".format(s)
            )
    elif s == "random":
        return random.randint(0, 9999)
    else:
        raise argparse.ArgumentTypeError(
            "Integer value is expected. Recieved {0}".format(s)
        )

def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, visual, acoustic,hcf,label_id): # an extra hcf
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.visual = visual
        self.acoustic = acoustic
        self.hcf = hcf
        self.label_id = label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    pop_count = 0
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) == 0:
            tokens_b.pop()
        else:
            pop_count += 1
            tokens_a.pop(0)
    return pop_count

#albert tokenizer split words in to subwords. "_" marker helps to find thos sub words
#our acoustic and visual features are aligned on word level. So we just create copy the same 
#visual/acoustic vectors that belong to same word.
def get_inversion(tokens, SPIECE_MARKER="_"):
    inversion_index = -1
    inversions = []
    for token in tokens:
        if SPIECE_MARKER in token:
            inversion_index += 1
        inversions.append(inversion_index)
    return inversions


def convert_humor_to_features(examples, tokenizer, punchline_only=False):
    features = []

    for (ex_index, example) in enumerate(examples):
        
        #p denotes punchline, c deontes context
        #hid is the utterance unique id. these id's are provided by the authors of urfunny and mustard
        #label is either 1/0 . 1=humor, 0=not humor
        (
            (p_words, p_visual, p_acoustic, p_hcf),# 
            (c_words, c_visual, c_acoustic, c_hcf),# 
            hid,
            label
        ) = example
                
        text_a = ". ".join(c_words)
        text_b = p_words + "."
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        
        inversions_a = get_inversion(tokens_a) #
        inversions_b = get_inversion(tokens_b) # 

        pop_count = _truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length - 3)

        inversions_a = inversions_a[pop_count:] # 
        inversions_b = inversions_b[: len(tokens_b)] # 

        visual_a = []
        acoustic_a = []
        hcf_a=[]        
        #our acoustic and visual features are aligned on word level. So we just 
        #create copy of the same visual/acoustic vectors that belong to same word.
        #because ber tokenizer split word into subwords
        for inv_id in inversions_a:
            visual_a.append(c_visual[inv_id, :])
            acoustic_a.append(c_acoustic[inv_id, :])
            hcf_a.append(c_hcf[inv_id, :])
            
        visual_a = np.array(visual_a)
        acoustic_a = np.array(acoustic_a)
        hcf_a = np.array(hcf_a)
        
        visual_b = []
        acoustic_b = []
        hcf_b = []
        for inv_id in inversions_b:
            visual_b.append(p_visual[inv_id, :])
            acoustic_b.append(p_acoustic[inv_id, :])
            hcf_b.append(p_hcf[inv_id, :])
        
        visual_b = np.array(visual_b)
        acoustic_b = np.array(acoustic_b)
        hcf_b = np.array(hcf_b)
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]

        acoustic_zero = np.zeros((1, ACOUSTIC_DIM_ALL))
        if len(tokens_a) == 0:
            acoustic = np.concatenate(
                (acoustic_zero, acoustic_zero, acoustic_b, acoustic_zero)
            )
        else:
            acoustic = np.concatenate(
                (acoustic_zero, acoustic_a, acoustic_zero, acoustic_b, acoustic_zero)
            )

        visual_zero = np.zeros((1, VISUAL_DIM_ALL))
        if len(tokens_a) == 0:
            visual = np.concatenate((visual_zero, visual_zero, visual_b, visual_zero))
        else:
            visual = np.concatenate(
                (visual_zero, visual_a, visual_zero, visual_b, visual_zero)
            )
        
        
        hcf_zero = np.zeros((1,4))
        if len(tokens_a) == 0:
            hcf = np.concatenate((hcf_zero, hcf_zero, hcf_b, hcf_zero))
        else:
            hcf = np.concatenate(
                (hcf_zero, hcf_a, hcf_zero, hcf_b, hcf_zero)
                
            )
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_mask = [1] * len(input_ids)
            
        acoustic_padding = np.zeros(
            (args.max_seq_length - len(input_ids), acoustic.shape[1])
        )
        acoustic = np.concatenate((acoustic, acoustic_padding))
        #original urfunny acoustic feature dimension is 81.
        #we found many features are highly correllated. so we removed
        #highly correlated feature to reduce dimension
        acoustic=np.take(acoustic, acoustic_features_list,axis=1)
        
        visual_padding = np.zeros(
            (args.max_seq_length - len(input_ids), visual.shape[1])
        )
        visual = np.concatenate((visual, visual_padding))
        #original urfunny visual feature dimension is more than 300.
        #we only considred the action unit and face shape parameter features
        visual = np.take(visual, visual_features_list,axis=1)
        
        
        hcf_padding= np.zeros(
            (args.max_seq_length - len(input_ids), hcf.shape[1])
        )
        
        hcf = np.concatenate((hcf, hcf_padding))
        
        padding = [0] * (args.max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length
        assert hcf.shape[0] == args.max_seq_length 
        
        label_id = float(label)
        
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                hcf=hcf,
                label_id=label_id,
            )
        )
            
    return features


def get_appropriate_dataset(data, tokenizer, parition):
    

    features = convert_humor_to_features(data, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    hcf = torch.tensor([f.hcf for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        hcf, # all_label_ids, just add a hcf, we can just remove the hcf 
        all_label_ids,
    )
    
    return dataset


def set_up_data_loader():
    if args.dataset=="urfunny":
        data_file = "ur_funny.pkl"
    elif args.dataset=="sarcasm":
        data_file = "mustard.pkl"
        
    with open(
        os.path.join(DATASET_LOCATION, data_file),
        "rb",
    ) as handle:
        all_data = pickle.load(handle)
        
    train_data = all_data["train"]
    dev_data = all_data["dev"]
    test_data = all_data["test"]
    tokenizer = BertTokenizer.from_pretrained('../BERT_EN/')

    train_dataset = get_appropriate_dataset(train_data, tokenizer, "train")
    dev_dataset = get_appropriate_dataset(dev_data, tokenizer, "dev")
    test_dataset = get_appropriate_dataset(test_data, tokenizer, "test")
    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=1
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True, num_workers=1
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=1
    )
    
    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )



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


def prep_for_training(num_train_optimization_steps: int):

    if args.model == "bert-base-uncased":
        model = GLoMo.from_pretrained(
            '../BERT_EN/', num_labels=2, args = args,
        )

   

    total_para = 0
    for param in model.parameters():
        total_para += np.prod(param.size())
    print('total parameter for the model: ', total_para)
    
    if args.load:
        model.load_state_dict(torch.load(args.model_path))

    model.to(DEVICE)

    return model

def adjust_learning_rate(optimizer, epoch, args):# 
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups: # 
        param_group['lr'] = lr
        
def train_epoch(model: nn.Module, train_dataloader: DataLoader, epoch=None):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # 
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    adjust_learning_rate(optimizer, epoch, args) 
    
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, hcf, label_ids = batch # for urfunny and sarcasm, it is input_ids, acoustic, visual, mask, segment. hcf, label_ids
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        moe_losses, logits, outputs = model(
            input_ids,
            visual,
            acoustic,
            label_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
        )

        loss_fct = CrossEntropyLoss()# BCEWithLogitsLoss
        loss_all = loss_fct(logits.view(-1, 2), label_ids.type(torch.cuda.LongTensor).view(-1)) + moe_losses
        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        if args.gradient_accumulation_step > 1:
            loss = loss_all / args.gradient_accumulation_step

        tr_loss += loss_all.item()
        nb_tr_steps += 1

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_steps = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, hcf, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            logits, outputs = model.test(
                input_ids,
                 visual,
                 acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
               # labels=None,
            )


            loss_fct = CrossEntropyLoss()# BCEWithLogitsLoss
            loss = loss_fct(logits.view(-1, 2), label_ids.type(torch.cuda.LongTensor).view(-1)) # binary classification
            
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, hcf, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            logits, outputs = model.test(
                input_ids,
                 visual,
                 acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

            # logits = outputs

            logits = np.argmax(logits.detach().cpu().numpy(), axis=1)
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    test_preds, test_truth = test_epoch(model, test_dataloader)
    acc2 =  accuracy_score(test_preds, test_truth)
    f1_weighted = f1_score(test_preds, test_truth, average='weighted')
    f1_average = f1_score(test_preds, test_truth, average='micro')

    return acc2, f1_weighted, f1_average


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader
):
    valid_losses = []
    test_accuracies = []
    f1_score = []
    best_loss = 9e+9
    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, epoch_i)
        valid_loss = eval_epoch(model, validation_dataloader)
        acc2, f1_weighted, f1_average = test_score_model(
            model, test_data_loader
        )

        print(
            "epoch:{}, train_loss:{:.4f}, valid_loss:{:.4f}, test_acc:{:.4f}".format(
                epoch_i, train_loss, valid_loss, acc2
            )
        )


        print(
            "current acc2:{:.4f}, f_weighted:{:.4f}, f_average:{:.4f}".format(
                acc2, f1_weighted, f1_average
            )
        )



def main():
    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model = prep_for_training(
        num_train_optimization_steps)#

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader
    )


if __name__ == "__main__":
    main()


