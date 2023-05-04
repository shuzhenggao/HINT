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
from my_lib import Example, read_examples, InputFeatures, convert_examples_to_features
import editdistance
import multiprocessing
from gensim import corpora
from gensim.summarization import bm25

train_data = []
with jsonlines.open('/resource/dataset/atlas_assert/Datasets/Raw_Dataset/train_ssl_processed.jsonl') as f:
    for obj in f:
        train_data.append(obj)


def process(raw_data):
    processed = []
    for idx in tqdm(range(len(raw_data))):
        obj = raw_data[idx]['conf']
        query = ' '.join(obj['code_tokens'])
        score = bm25_model.get_scores(query,average_idf)
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[0]
        code_candidates_tokens = []
        obj['dist_id'] = rtn[0]
        processed.append(obj)
    return processed    

def pseudo_labeling(model, args, tokenizer, device):
    if not os.path.exists(os.path.join(args.output_dir, 'pseudo.jsonl')):
        #generate pseudo label
        eval_examples = read_examples(args.unlabel_filename, args)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
        all_source_ids = eval_features['source_ids']
        all_source_mask = eval_features['source_mask']
        eval_data = TensorDataset(all_source_ids, all_source_mask)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=5)
        model.eval()
        generated_texts = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                #do_sample=True, top_p=top_p, temperature=temperature
                generated_texts_ids = model.generate(input_ids=source_ids, attention_mask=source_mask,num_beams=args.beam_size,
                                                                       max_length=args.max_target_length)
                for text_ids in generated_texts_ids:
                    text = tokenizer.decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    generated_texts.append(text)
        model.train()
        pseudo_data = []
        for idx in range(len(generated_texts)):
            pseudo_data.append({'code_tokens':eval_examples[idx].origin_source, 'docstring_tokens':generated_texts[idx].split(' ')})
        with jsonlines.open(os.path.join(args.output_dir, 'pseudo.jsonl'), mode='w') as f:
            f.write_all(pseudo_data)



    if not os.path.exists(os.path.join(args.output_dir, 'pseudo_conf1.jsonl')):
        #calculate loss
        eval_examples = read_examples(os.path.join(args.output_dir, 'pseudo.jsonl'), args)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
        all_source_ids = eval_features['source_ids']
        all_source_mask = eval_features['source_mask']
        all_target_ids = eval_features['target_ids']
        all_target_mask = eval_features['target_mask']
        eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1,num_workers=5)
        score = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch
            with torch.no_grad():
                labels = [
                        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for
                        labels_example in target_ids
                ]
                labels = torch.tensor(labels).to(device)
                tokens_num = torch.tensor([(labels_example != -100).sum().item() for labels_example in labels]).sum().item()
                loss = model(input_ids=source_ids, attention_mask=source_mask, labels=labels).loss
            score.append(loss.div(tokens_num).cpu().item())
        pseudo_data = []
        for idx in range(len(score)):
            pseudo_data.append({'code_tokens': eval_examples[idx].origin_source, 'docstring_tokens':eval_examples[idx].origin_target, 'loss': score[idx]})
        with jsonlines.open(os.path.join(args.output_dir, 'pseudo_conf1.jsonl'), mode='w') as f:
            f.write_all(pseudo_data)



    #preprocess
    pseudo_data = []
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_conf1.jsonl')) as f:
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
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_conf1.jsonl'), mode='w') as f:
        f.write_all(processed_results)

    '''pseudo_data = []
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_conf1.jsonl')) as f:
        for i in f:
            pseudo_data.append(i)
    distance_data = []
    with jsonlines.open('pseudo_dis_conf.jsonl') as f:
        for i in f:
            distance_data.append(i)
    for i,j in zip(pseudo_data,distance_data):
        assert ' '.join(i['code_tokens']) == ' '.join(j['code_tokens'])
        i['dist_id'] = j['dist_id']
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_dis_conf.jsonl'), mode='w') as f:
        f.write_all(pseudo_data)'''



    #confidence filter bad samples by loss and involve good samples by knn
    train_data = []
    with jsonlines.open(args.train_file) as f:
        for obj in f:
            train_data.append(obj)
    pseudo_data = []
    total_loss = []
    with jsonlines.open(os.path.join(args.output_dir, 'pseudo_dis_conf.jsonl')) as f:
        for i in f:
            pseudo_data.append(i)
            total_loss.append(i['loss'])
    total_loss = sorted(total_loss)
    threshold = total_loss[int(len(total_loss)*args.percent)]
    print('threshold:',threshold)

    selected_pseudo_data = []
    for obj in tqdm(pseudo_data):
        dist_id = obj['dist_id']
        source_norm = editdistance.eval(train_data[dist_id]['code_tokens'], obj['code_tokens']) / len(obj['code_tokens'])
        target_norm = editdistance.eval(train_data[dist_id]['docstring_tokens'], obj['docstring_tokens']) / len(obj['docstring_tokens'])
        s=args.threshold
        if source_norm<=s and target_norm<=s:
            selected_pseudo_data.append(obj)
        elif source_norm<=s and target_norm>=1-s:
            continue
        if (obj['loss'])<threshold:
            selected_pseudo_data.append(obj)
    with jsonlines.open(os.path.join(args.output_dir, 'selected_pseudo.jsonl'), mode='w') as f:
        f.write_all(selected_pseudo_data)
    print('number:',len(selected_pseudo_data))


    pass


