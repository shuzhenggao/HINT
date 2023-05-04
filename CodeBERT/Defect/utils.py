import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from math import log

import numpy as np
import torch
import copy
import heapq
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
import jsonlines
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import torch.nn.functional as F
import multiprocessing
from model import Model
from sklearn.metrics import classification_report
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
import editdistance
import multiprocessing


train_data = []
with jsonlines.open('train.jsonl') as f:
    for obj in f:
        train_data.append(obj)
code = [' '.join(obj['code_tokens']) for obj in train_data]
bm25_model = bm25.BM25(code)
average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())



def process(raw_data):
    processed = []
    for idx in tqdm(range(len(raw_data))):
        obj = raw_data[idx]['conf']
        query = ' '.join(obj['code_tokens'])
        score = bm25_model.get_scores(query,average_idf)
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[0]
        obj['dist_id'] = rtn[0]
        processed.append(obj)
    return processed    


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,
                 score,
                 origin_source,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label
        self.score=score
        self.origin_source=origin_source

        
def convert_examples_to_features(js,tokenizer,args):
    code=' '.join(js['code_tokens'])
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    #if (args.do_train and 'score' in js and js['score']<1):
    #    for index in range(len(source_ids)):
    #        prob = random.random()
    #        if prob < 0.03:
    #            source_ids[index] = tokenizer.mask_token_id
            #elif prob < 0.05:
            #    source_ids[index] = random.choice(vacab_list)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length

    if 'score' not in js:
        score = 1.0
    else:
        score=1.0
    #label = int(js['origina_label']) if 'origina_label' in js else int(js['label'])
    label = int(js['label'])

    return InputFeatures(source_tokens,source_ids,0,label,float(score),js['code_tokens'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, teacher_model=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.mode = True if 'train' in file_path else False
        if teacher_model is not None:
            self.config = teacher_model.config
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'pl_ours' in args.mode and 'train' in file_path:
            pseudo_labeling(teacher_model, args, tokenizer, args.device)
            with open(os.path.join(args.output_dir, 'selected_pseudo.jsonl')) as f:
                for line in f:
                    js=json.loads(line.strip())
                    self.examples.append(convert_examples_to_features(js,tokenizer,args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        '''input_ids =  copy.deepcopy(self.examples[i].input_ids)
        if True or self.mode and self.examples[i].score<1:
            length = sum([obj != self.tokenizer.pad_token_id for obj in input_ids]) 
            pos = random.sample(range(length),length//20)
            for idx in pos:
                sub_idx = random.randint(3, self.config.vocab_size-1)
                input_ids[idx] = sub_idx
                #input_ids[idx] = self.tokenizer.mask_token_id'''
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].score)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def pseudo_labeling(model, args, tokenizer, device):
    if not os.path.exists(os.path.join(args.output_dir, 'pseudo.jsonl')):
        eval_dataset = TextDataset(tokenizer, args, args.unlabel_filename)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
        model.eval()
        logits=[] 
        probs=[]
        for batch in eval_dataloader:
            inputs = batch[0].to(args.device)        
            label = batch[1].to(args.device) 
            score = batch[2].to(args.device)
            with torch.no_grad():
                lm_loss,logit = model(inputs,label,score,inputs)
                logits.append(logit.cpu().numpy())
                probs.append((torch.sum(F.softmax(logit, dim=-1).clamp(0.5,1), -1)-0.5).cpu().numpy())
        logits = np.concatenate(logits,0)
        probs = np.concatenate(probs,0)
        preds = np.argmax(logits, axis=-1)
    
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        pseudo_data = []
        selected_pseudo_data = []
        for idx in range(len(preds)):
            pseudo_data.append({'code_tokens':eval_dataset.examples[idx].origin_source, 'label':int(preds[idx]), 'origina_label':eval_dataset.examples[idx].label, 'logit':logits[idx].tolist(), 'score': float(probs[idx])})
        with jsonlines.open(os.path.join(args.output_dir, 'pseudo.jsonl'), mode='w') as f:
            f.write_all(pseudo_data)

    #preprocess
    pseudo_data = []
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo.jsonl')) as f:
        for i in f:
            pseudo_data.append(i)

    processes = int(multiprocessing.cpu_count()*0.8)
    pool = multiprocessing.Pool(processes)
    data = [{'conf':pseudo_data[idx]} for idx in range(len(pseudo_data))]
    chunk_size = len(data) // processes
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    results = pool.map(process, chunks)
    processed_results = []
    for chunk_results in results:
        for res in chunk_results:
            processed_results.append(res)
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_dis.jsonl'), mode='w') as f:
        f.write_all(processed_results)

    '''f not os.path.exists(os.path.join(args.output_dir, 'pseudo_dis.jsonl')):
        pseudo_data = []
        with jsonlines.open(os.path.join(args.output_dir, 'pseudo.jsonl')) as f:
            for i in f:
                pseudo_data.append(i)
        distance_data = []
        with jsonlines.open('../defect_pl_ours/pseudo_dis.jsonl') as f:
            for i in f:
                distance_data.append(i)
        for i,j in zip(pseudo_data,distance_data):
            assert ' '.join(i['code_tokens']) == ' '.join(j['code_tokens'])
            i['dist_id'] = j['dist_id']
        with jsonlines.open(os.path.join(args.output_dir, 'pseudo_dis.jsonl'), mode='w') as f:
            f.write_all(pseudo_data)'''


    #confidence filter bad samples by loss and involve good samples by knn
    train_data = []
    with jsonlines.open(args.train_data_file) as f:
        for obj in f:
            train_data.append(obj)
    pseudo_data = []
    pos_total_loss = []
    neg_total_loss = []
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_dis.jsonl')) as f:
        for i in f:
            pseudo_data.append(i)
            if int(i['label'])==1:
                pos_total_loss.append(-log(i['score']))
            else:
                neg_total_loss.append(-log(i['score']))
    pos_total_loss = sorted(pos_total_loss)
    neg_total_loss = sorted(neg_total_loss)
    pos_threshold = pos_total_loss[int(len(pos_total_loss)*args.threshold)]
    neg_threshold = neg_total_loss[int(len(neg_total_loss)*args.threshold)]
    print('pos_threshold:',pos_threshold)
    print('neg_threshold:',neg_threshold)

    pos_selected_pseudo_data = []
    neg_selected_pseudo_data = []
    for obj in tqdm(pseudo_data):
        dist_id = obj['dist_id']
        source_norm = editdistance.eval(train_data[dist_id]['code_tokens'], obj['code_tokens']) / len(obj['code_tokens'])
        s=0.4
        if int(obj['label'])==1:
            loss_value = -log(obj['score'])
            if source_norm<=s and int(obj['label']) == int(train_data[dist_id]['label']):
                pos_selected_pseudo_data.append(obj)
            elif source_norm<=s and int(obj['label']) != int(train_data[dist_id]['label']):
                continue
            elif loss_value<pos_threshold:
                pos_selected_pseudo_data.append(obj)
        else:
            loss_value = -log(obj['score'])
            if source_norm<=s and int(obj['label']) == int(train_data[dist_id]['label']):
                neg_selected_pseudo_data.append(obj)
            elif source_norm<=s and int(obj['label']) != int(train_data[dist_id]['label']):
                continue
            elif loss_value<neg_threshold:
                neg_selected_pseudo_data.append(obj)
    pos_number = len(pos_selected_pseudo_data)
    neg_number = len(neg_selected_pseudo_data)
    selected_pseudo_data = random.sample(pos_selected_pseudo_data, k=int(min(pos_number,neg_number/16.3))) + random.sample(neg_selected_pseudo_data, k=int(min(pos_number*16.3,neg_number)))
    
    
    print(pos_number,neg_number,len(selected_pseudo_data),len(selected_pseudo_data)/len(pseudo_data))
    with jsonlines.open(os.path.join(args.output_dir, 'selected_pseudo.jsonl'), mode='w') as f:
        f.write_all(selected_pseudo_data)

    '''pos_selected_pseudo_data = []
    neg_selected_pseudo_data = []
    pos_score = []
    neg_score = []
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo.jsonl')) as f:
        for obj in f:
            if int(obj['label'])==1 and obj['score']>0.8:
                pos_selected_pseudo_data.append(obj)
                pos_score.append(obj['score'])
            elif int(obj['label'])==0 and obj['score']>0.99:
                neg_selected_pseudo_data.append(obj)
                neg_score.append(obj['score'])
    pos_number = len(pos_score)
    neg_number = len(neg_score)
    #selected_pseudo_data = random.sample(pos_selected_pseudo_data, k=50000) + random.sample(neg_selected_pseudo_data, k=40000)
    selected_pseudo_data = pos_selected_pseudo_data + random.sample(neg_selected_pseudo_data, k=16*pos_number)
    #pos_topk = heapq.nlargest(min(pos_number,neg_number), range(len(pos_score)), pos_score.__getitem__)
    #neg_topk = heapq.nlargest(min(pos_number,neg_number), range(len(neg_score)), neg_score.__getitem__)
    #selected_pseudo_data = []
    #for idx in pos_topk:
    #    selected_pseudo_data.append(pos_selected_pseudo_data[idx])
    #for idx in neg_topk:
    #    selected_pseudo_data.append(neg_selected_pseudo_data[idx])


    print(pos_number, neg_number, neg_number/pos_number , args.threshold)
    print(len(selected_pseudo_data), " samples are selected")
    with jsonlines.open(os.path.join(args.output_dir, 'selected_pseudo.jsonl'), mode='w') as f:
        f.write_all(selected_pseudo_data)
    pass'''
    
    

