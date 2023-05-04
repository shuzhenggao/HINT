import os
import sys
import math
import numpy as np
import torch
import argparse
import json
import random
import jsonlines
import heapq
import pandas as pd
from tqdm import tqdm, trange
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset

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
            target_tokens = tokenizer.tokenize("None")
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

def pseudo_labeling(model, args, tokenizer, device):
    eval_examples = read_examples(args.unlabel_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids)
    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    outputs=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids= batch[0]
        with torch.no_grad():
            preds = model(source_ids=source_ids)
            flag=1
            for pred in preds:
                output=pred[0].cpu().numpy()
                output=list(output)
                if 0 in output:
                    pos = output.index(0)
                    output=output[:pos]
                output = tokenizer.decode(output,clean_up_tokenization_spaces=False)
                outputs.append(output)
    
    pseudo_data = []
    for idx in range(len(outputs)):
        pseudo_data.append({'code_tokens':eval_examples[idx].origin_source, 'docstring_tokens':outputs[idx].split(' ')})
    with jsonlines.open(args.label_filename, mode='w') as f:
        f.write_all(pseudo_data)
    
    #os.system('bash ../Unixcoder/sample_selection.sh '+str(args.lang)+' '+str(args.threshold)+' '+str(args.output_dir))
    

