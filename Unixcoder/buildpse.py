import os
import sys
import math
import numpy as np
import torch
import argparse
import random
import jsonlines
import json
import heapq
import pandas as pd
from tqdm import tqdm, trange
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import editdistance
import heapq
import time
class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 origin_target,
                 origin_source,
                 score
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.origin_target = origin_target
        self.origin_source = origin_source
        self.score = score

def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code = ' '.join(js['code_tokens']).replace('\n',' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n','')
            nl = ' '.join(nl.strip().split())            
            examples.append(
                Example(
                        idx = idx,
                        source = code,
                        target = nl,
                        origin_source=js['code_tokens'],
                        origin_target=js['docstring_tokens'],
                        score=js['score'] if 'score' in js else 1,
                        ) 
            )
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 score
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.score = score      
        
def convert_examples_to_features(examples, tokenizer, args,stage=None):
    """convert examples to token ids"""
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-5]
        source_tokens = [tokenizer.cls_token,"<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens) 
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length

        #target
        if stage=="test":
            #target_tokens = tokenizer.tokenize("None")
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = ["<mask0>"] + target_tokens + [tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        score = [example.score]*args.max_target_length
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 score
            )
        )
    return features

def build_pse(model, args, tokenizer, device):
    eval_examples = read_examples(args.label_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)    
    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)    
    all_score = torch.tensor([f.score for f in eval_features], dtype=torch.float)     
    eval_data = TensorDataset(all_source_ids,all_target_ids,all_score)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1,num_workers=5)
    score = []
    token_score = []
    args.mode='bsl'
    for batch in tqdm(eval_dataloader):
      batch = tuple(t.to(device) for t in batch)
      source_ids,target_ids,s = batch                  
      with torch.no_grad():
          _,loss,num = model(source_ids=source_ids,
                              target_ids=target_ids,score=s,args=args)
      score.append(loss.cpu().item())
      token_score.append(loss.div(num).cpu().item())
    dist_data=[]
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_conf2_all.jsonl')) as f:
        for i in f:
            dist_data.append(i)
    pseudo_data = []
    with jsonlines.open(args.label_filename) as f:
        for i in f:
            pseudo_data.append(i)       
    for idx in range(len(score)):
        pseudo_data[idx]['loss1']=score[idx]
        pseudo_data[idx]['loss2']=token_score[idx]
        pseudo_data[idx]['dist_id']=dist_data[idx]['dist_id']
    with jsonlines.open(os.path.join(args.output_dir, args.pseudo_filename), mode='w') as f:
        f.write_all(pseudo_data)
    '''
    #confidence filter bad samples by loss and involve good samples by knn
    train_data = []
    with jsonlines.open(args.train_filename) as f:
        for obj in f:
            train_data.append(obj)
    selected_pseudo_data = []
    pseudo_data = []
    with jsonlines.open(os.path.join(args.output_dir, args.pseudo_filename)) as f:
        for i in f:
            pseudo_data.append(i)
    topk = heapq.nsmallest(round(len(pseudo_data)*args.threshold), pseudo_data, key=lambda s: s['loss2'])
    print(args.selected_pseudo_filename)
    for obj in tqdm(pseudo_data):
        dist_id = obj['dist_id']
        source_norm = editdistance.eval(train_data[dist_id]['code_tokens'], obj['code_tokens']) / len(obj['code_tokens'])
        target_norm = editdistance.eval(train_data[dist_id]['docstring_tokens'], obj['docstring_tokens']) / len(obj['docstring_tokens'])
        s=args.s
        if source_norm<=s and target_norm<=s:
            selected_pseudo_data.append(obj)
        elif source_norm<=s and target_norm>=1-s:
            continue
        elif obj in topk:
        #elif (obj['loss2'])<args.threshold:
            selected_pseudo_data.append(obj)
    with jsonlines.open(args.selected_pseudo_filename, mode='w') as f:
        f.write_all(selected_pseudo_data)
    '''
    pass
