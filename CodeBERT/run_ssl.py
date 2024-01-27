from __future__ import absolute_import
import os
import sys
import bleu
import evaluation
import pickle
import torch
import json
import random
import time
import copy
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from selectnow import select_pseudo
from utils import Example, read_examples, InputFeatures, convert_examples_to_features
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
 
def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
               
def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization") 
    ## Parameters of our method
    parser.add_argument("--unlabel_filename", default=None, type=str, 
                        help="The unlabel filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--mode", default="", type=str,
                        help="Mode")
    parser.add_argument("--threshold", default=0.8, type=float, 
                        help="The threshold of pseudo labeling data.")
    parser.add_argument("--lang", default="", type=str,
                        help="The Language.")
    parser.add_argument("--out_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--pseudo_filename", default=None, type=str, 
                        help="The pseudo filename. Should contain the .jsonl files for this task.")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    # make dir if log not exist 
    log_dir='log/'+args.lang 
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    if 'pl_ours' in args.mode:
        config.attention_probs_dropout_prob = 0.1
        config.hidden_dropout_prob = 0.1
    
    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,args=args,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        if args.do_test:
            model.load_state_dict(torch.load(args.load_model_path))
        else:
            teacher_encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
            teacher_decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
            teacher_decoder = nn.TransformerDecoder(teacher_decoder_layer, num_layers=6)
            teacher_model=Seq2Seq(encoder=teacher_encoder,decoder=teacher_decoder,config=config,
                          beam_size=args.beam_size,max_length=args.max_target_length,
                          sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
            teacher_model.load_state_dict(torch.load(args.load_model_path))
            teacher_model.to(device)
            if args.local_rank != -1:
                teacher_model = DDP(teacher_model)
            elif args.n_gpu > 1:
                teacher_model = torch.nn.DataParallel(teacher_model)

        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        log_name='train_ssl.txt'
        with open(os.path.join(log_dir,log_name), 'a+') as f:
          now=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
          f.write(now)
          f.write("\n")
          f.write(str(args))
          f.write('\n[model]:{}\n'.format(args.output_dir+'/'+args.out_dir))
        train_examples = read_examples(args.train_filename)
        if 'pl_ours' in args.mode:
            origin_train_examples = copy.deepcopy(train_examples)
            select_pseudo(teacher_model, args, tokenizer, device)
            train_pseudo_examples = read_examples(args.pseudo_filename)
            train_examples = origin_train_examples + train_pseudo_examples
            print(len(train_pseudo_examples))
            print(len(origin_train_examples))
            print(len(origin_train_examples)+len(train_pseudo_examples))
            print(len(train_examples))

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=(len(train_examples))*args.num_train_epochs)
            
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0,0,0,0,0,1e6
        train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)   
        all_score = torch.tensor([f.score for f in train_features], dtype=torch.float)     
        train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask,all_score)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        for epoch in range(args.num_train_epochs):
            print('Epoch: ',epoch)
            bar = tqdm(train_dataloader,total=len(train_dataloader))

            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,target_ids,target_mask,score = batch
                loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask,score=score,args=args)
                
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("loss {}".format(train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
    
                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval:                            
                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples,min(100000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids,all_source_mask)
                    dev_dataset['dev_bleu']=eval_examples,eval_data
                
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask= batch                  
                    with torch.no_grad():
                        preds, _ = model(source_ids=source_ids,source_mask=source_mask)  
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')

                if(args.lang=="assert"):
                    gold_path=os.path.join(args.output_dir, "dev.gold")
                    pred_path=os.path.join(args.output_dir, "dev.output")
                    metric1,metric2,metric3,metric4=evaluation.cal_match(gold_path,pred_path)
                    logger.info("  %s = %s "%("exact_match_count*100/len(pred)",str(metric1)))
                    logger.info("  %s = %s "%("is_match_count*100/len(pred)",str(metric2)))
                    logger.info("  %s = %s "%("lcs*100/len(pred)",str(metric3)))
                    logger.info("  %s = %s "%("ed/len(pred)",str(metric4)))
                    logger.info("  "+"*"*20) 
                    if metric1>best_bleu:
                        logger.info("  Best exact_match:%s",metric1)
                        logger.info("  "+"*"*20)
                        best_bleu=metric1
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, args.out_dir)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        best_model = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(best_model.state_dict(), output_model_file)
                else:
                    (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
                    dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                    logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                    logger.info("  "+"*"*20)  
                    if dev_bleu>best_bleu:
                        logger.info("  Best bleu:%s",dev_bleu)
                        logger.info("  "+"*"*20)
                        best_bleu=dev_bleu
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, args.out_dir)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        best_model = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(best_model.state_dict(), output_model_file)
                    with open(os.path.join(log_dir,log_name), 'a+') as f:
                        f.write('[epoch: {}] \n'.format(str(epoch)))
                        f.write('bleu: {} best bleu: {}\n'.format(str(dev_bleu),str(best_bleu)))
                        if(epoch==9):
                            f.write('[Time: {}] \n'.format(get_elapse_time(t0)))

               
    if args.do_test:
        log_name='test_ssl.txt'
        with open(os.path.join(log_dir,log_name), 'a+') as f:
            now=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            f.write(now)
            f.write('\n')
            f.write(str(args))
        files=[]
        if args.test_filename is not None:
            for file_name in args.test_filename.split(','):
                files.append(file_name)
        for idx,file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
            eval_data = TensorDataset(all_source_ids,all_source_mask)   

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval() 
            p=[]
            scores=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask= batch                  
                with torch.no_grad():
                    preds, confs = model(source_ids=source_ids,source_mask=source_mask)
                    for pred, conf in zip(preds,confs):
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        scores.append(conf[0])
                        p.append(text)
        
            model.train()
            predictions=[]
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                for ref,score,gold in zip(p,scores,eval_examples):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(str(gold.idx)+'\t'+ref+'\t'+str(score)+'\n')
                    f1.write(str(gold.idx)+'\t'+gold.target+'\n')     
            if(args.lang=="assert"):
                gold_path=os.path.join(args.output_dir, "test_0.gold")
                pred_path=os.path.join(args.output_dir, "test_0.output")
                metric1,metric2,metric3,metric4=evaluation.cal_match(gold_path,pred_path)
                logger.info("  %s = %s "%("exact_match_count*100/len(pred)",str(metric1)))
                logger.info("  %s = %s "%("is_match_count*100/len(pred)",str(metric2)))
                logger.info("  %s = %s "%("lcs*100/len(pred)",str(metric3)))
                logger.info("  %s = %s "%("ed/len(pred)",str(metric4)))
                logger.info("  "+"*"*20) 
            else:
                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_{}.gold".format(idx))) 
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)
                with open(os.path.join(log_dir,log_name), 'a+') as f:
                  f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), args.load_model_path))
                  f.write(str(dev_bleu))
                  f.write('\n')
            
        
        '''
        #beam search in T5
        generated_texts = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                generated_texts_ids = model.generate(input_ids=source_ids, attention_mask=source_mask,num_beams=args.beam_size,
                                                     max_length=args.max_target_length)
                for text_ids in generated_texts_ids:
                    text = tokenizer.decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    generated_texts.append(text)'''
                
                
if __name__ == "__main__":
    main()


