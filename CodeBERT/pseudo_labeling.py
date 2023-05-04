import os
import sys
import math
import numpy as np
import torch
import argparse
import random
import jsonlines
import heapq
import pandas as pd
from tqdm import tqdm, trange
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from utils import Example, read_examples, InputFeatures, convert_examples_to_features

def pseudo_labeling(model, args, tokenizer, device):
    eval_examples = read_examples(args.unlabel_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids,all_source_mask)
    # Calculate bleu
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    outputs=[]
    scores=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask= batch
        with torch.no_grad():
            preds, confs = model(source_ids=source_ids,source_mask=source_mask)
            for pred,conf in zip(preds,confs):
                output=pred[0].cpu().numpy()
                output=list(output)
                if 0 in output:
                    pos = output.index(0)
                    output=output[:pos]
                output = tokenizer.decode(output,clean_up_tokenization_spaces=False)
                outputs.append(output)
                scores.append(conf[0])
    
    pseudo_data = []
    for idx in range(len(outputs)):
        pseudo_data.append({'code_tokens':eval_examples[idx].origin_source, 'docstring_tokens':outputs[idx].split(' ')})
    with jsonlines.open(args.label_filename, mode='w') as f:
        f.write_all(pseudo_data)
    
    #os.system('bash ../Unixcoder/sample_selection.sh '+str(args.lang)+' '+str(args.threshold)+' '+str(args.output_dir))
    

