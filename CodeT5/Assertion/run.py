from __future__ import absolute_import
import os
import time
import copy
from pseudo_labeling import pseudo_labeling

import torch
import random
import logging
import argparse
import numpy as np
from io import open
from tqdm import tqdm
from random import choice
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, T5Config, T5ForConditionalGeneration)

import bleu
from my_lib import read_examples, convert_examples_to_features, get_elapse_time, TextDataset

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}


ruby_special_token = ['keyword', 'identifier', 'separators', 'simple_symbol', 'constant', 'instance_variable',
 'operator', 'string_content', 'integer', 'escape_sequence', 'comment', 'hash_key_symbol',
  'global_variable', 'heredoc_beginning', 'heredoc_content', 'heredoc_end', 'class_variable',]

java_special_token = ['keyword', 'identifier', 'type_identifier',  'separators', 'operator', 'decimal_integer_literal',
 'void_type', 'string_literal', 'decimal_floating_point_literal', 
 'boolean_type', 'null_literal', 'comment', 'hex_integer_literal', 'character_literal']

go_special_token = ['keyword', 'identifier', 'separators', 'type_identifier', 'int_literal', 'operator', 
'field_identifier', 'package_identifier', 'comment',  'escape_sequence', 'raw_string_literal',
'rune_literal', 'label_name', 'float_literal']

javascript_special_token =['keyword', 'separators', 'identifier', 'property_identifier', 'operator', 
'number', 'string_fragment', 'comment', 'regex_pattern', 'shorthand_property_identifier_pattern', 
'shorthand_property_identifier', 'regex_flags', 'escape_sequence', 'statement_identifier']

php_special_token =['text', 'php_tag', 'name', 'operator', 'keyword', 'string', 'integer', 'separators', 'comment', 
'escape_sequence', 'ERROR',  'boolean', 'namespace', 'class', 'extends']

python_special_token =['keyword', 'identifier', 'separators', 'operator', '"', 'integer', 
'comment', 'none', 'escape_sequence']


special_token={
    'python':python_special_token,
    'java':java_special_token,
    'ruby':ruby_special_token,
    'go':go_special_token,
    'php':php_special_token,
    'javascript':javascript_special_token
}

all_special_token = []
for key, value in special_token.items():
    all_special_token = list(set(all_special_token ).union(set(value)))

def mask_tokens(inputs,tokenizer,mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()] # for masking special token
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0) # masked padding
        
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(inputs.device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def replace_with_type_tokens(inputs,replaces,tokenizer,mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()] # for masking special token
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0) # masked padding
        
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = replaces[indices_replaced] 

    return inputs, labels

def replace_special_token_with_type_tokens(inputs, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,0.0).to(inputs.device)   
    probability_matrix.masked_fill_(labels.eq(speical_token_ids).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] =  speical_token_ids

    return inputs, labels

def replace_special_token_with_mask(inputs, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,0.0).to(inputs.device)   
    probability_matrix.masked_fill_(labels.eq(speical_token_ids).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] =tokenizer.convert_tokens_to_ids(tokenizer.mask_token) 

    return inputs, labels



class SCELoss(torch.nn.Module):
    def __init__(self, config, alpha=1, beta=1):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.config = config
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(self, pred, labels, score=None):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.config.vocab_size).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        if score is None:
            loss = (self.alpha * ce.mean())  + (self.beta * rce.mean())
        else:
            loss = (self.alpha * ce.mean())  + (self.beta * rce.mean())
            #loss = (score*ce).mean() + ((2-score)*rce).mean()
        
        return loss

def compute_kl_loss(p, q):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), torch.clamp(F.softmax(q, dim=-1), min=1e-7, max=1.0), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), torch.clamp(F.softmax(p, dim=-1), min=1e-7, max=1.0), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = torch.mean(torch.sum(p_loss, dim=1), dim=0)
    q_loss = torch.mean(torch.sum(q_loss, dim=1), dim=0)

    loss = (p_loss + q_loss)
    return loss


def label_smoothed_nll_loss(net_output, target, epsilon, ignore_index=None, reduce=True):
    #seq_len x hidden
    lprobs = F.log_softmax(net_output, dim=-1)
    lprobs = lprobs.view(-1, lprobs.size(-1))

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_arguments():
    parser = argparse.ArgumentParser()

    # outdated parameters
    parser.add_argument("--model_type", default=None, type=str, required=False,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name", default=None, type=str, required=False,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--train_file", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_file", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_file", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  

    # Required parameters
    #parser.add_argument("--log_name", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default="./model", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default="./data", type=str,
                        help="Path to the dir which contains processed data for some languages")
    parser.add_argument("--lang", default=None, type=str,
                        help="language to summarize")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--visible_gpu', type=str, default="",
                        help="use how many gpus")
    parser.add_argument("--add_task_prefix", default=False, action='store_true',
                        help="Whether to add task prefix for T5 and codeT5")
    parser.add_argument("--add_lang_ids", default=False, action='store_true',
                        help="Whether to add language prefix for T5 and codeT5")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # other arguments
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
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
    parser.add_argument("--warm_up_ratio", default=0.1, type=float)

    # controlling arguments
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--early_stop_threshold', type=int, default=10)
    ## Parameters of our method
    parser.add_argument("--unlabel_filename", default=None, type=str, 
                        help="The unlabel filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--mode", default="", type=str,
                        help="Mode")
    parser.add_argument("--percent", default=0.8, type=float, 
                        help="The percent of pseudo labeling data.")
    parser.add_argument("--threshold", default=0.8, type=float, 
                        help="The threshold of pseudo labeling data.")
    parser.add_argument("--k", default=2, type=float,
                        help="parameter")
    parser.add_argument("--mlm_probability", default=0.1, type=float, 
                        help="mlm_probability")

    # print arguments
    args = parser.parse_args()

    return args


def main(args):
    set_seed(args.seed)
    model_name = args.model_name
    # data path
    train_filename = args.train_file
    dev_filename = args.dev_file
    test_filename = args.test_file

    # Setup CUDA, GPU & distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, model_name: %s", 
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), model_name)

    args.device = device

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # *********************************************************************************************************

    # read model --------------------------------------------------------------
    model_config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, config=model_config)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    special_token_list = all_special_token 
    special_token_id_list = tokenizer.convert_tokens_to_ids(special_token_list)

    if args.do_test or ('pl_ours' in args.mode and args.load_model_path is not None):
        logger.info("reload model from {}".format(args.load_model_path))
        if args.do_test:
            model.load_state_dict(torch.load(args.load_model_path))
        else:
            teacher_model = T5ForConditionalGeneration.from_pretrained(model_name, config=model_config)
            teacher_model.load_state_dict(torch.load(args.load_model_path))
            teacher_model.to(device)
            if args.local_rank != -1:
                teacher_model = DDP(teacher_model)
            elif args.n_gpu > 1:
                teacher_model = torch.nn.DataParallel(teacher_model)

    model.to(device)

    # parallel or distribute setting
    if args.local_rank != -1:
        # Distributed training
        try:
            # from apex.parallel import DistributedDataParallel as DDP
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    logger.info("model created!")

    # train part --------------------------------------------------------------
    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(train_filename, args)
        if 'pl_ours' in args.mode:
            origin_train_examples = copy.deepcopy(train_examples)
            pseudo_labeling(teacher_model, args, tokenizer, device)
            train_pseudo_examples = read_examples(os.path.join(args.output_dir, 'selected_pseudo.jsonl'), args)
            #train_pseudo_examples = []
            train_examples = train_examples + train_pseudo_examples
        N = len(train_examples)
        logger.info("Total {} training instances ".format(len(train_examples)))
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')

        all_source_ids = train_features['source_ids']
        all_source_mask = train_features['source_mask']
        all_target_ids = train_features['target_ids']
        all_target_mask = train_features['target_mask']
        all_scores = train_features['all_scores']

        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask, all_scores)
        #train_data = TextDataset(tokenizer, args, model_config, train_features, True)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps,
                                      num_workers=5)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = ((len(train_dataloader))// args.gradient_accumulation_steps) * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * args.warm_up_ratio),
                                                    num_training_steps=t_total)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        # used to save tokenized data
        dev_dataset = {}
        best_model = model.module if hasattr(model, 'module') else model
        nb_tr_examples, nb_tr_steps, global_step, best_bleu, best_loss = 0, 0, 0, 0, 1e6
        early_stop_threshold = args.early_stop_threshold

        early_stop_count = 0
        for epoch in range(args.num_train_epochs):
            model.train()
            tr_loss = 0.0
            tr_kl_loss = 0.0
            train_loss = 0.0

            bar = tqdm(train_dataloader, total=len(train_dataloader))
            idx=0
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask, scores = batch

                source_ids_aug = source_ids.clone().detach()
                if args.mlm_probability>0:
                    idx = random.randint(0,3)
                    if idx%4 == 0:
                        source_ids_aug[:, 2:], _ = mask_tokens(source_ids.clone()[:, 2:] ,tokenizer,args.mlm_probability)
                    elif idx%4 == 1:
                        code_types = source_ids.clone()
                        source_ids_aug[:, 2:], _ = replace_with_type_tokens(source_ids.clone()[:, 2:], code_types.clone()[:, 2:],tokenizer,args.mlm_probability)
                    elif idx%4 == 2:
                        choice_token_id  = choice(special_token_id_list)
                        source_ids_aug[:, 2:], _ = replace_special_token_with_type_tokens(source_ids.clone()[:, 2:], choice_token_id, tokenizer,args.mlm_probability)
                    elif idx%4 == 3:
                        choice_token_id  = choice(special_token_id_list)
                        source_ids_aug[:, 2:], _ = replace_special_token_with_mask(source_ids.clone()[:, 2:], choice_token_id,tokenizer, args.mlm_probability)

                labels = [
                    [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for
                    labels_example in target_ids
                ]
                labels = torch.tensor(labels).to(device)

                out = model(input_ids=source_ids, attention_mask=source_mask, labels=labels, return_dict=True)
                if 'pl_ours' in args.mode:
                    loss_fct=SCELoss(model_config, alpha=1, beta=1)
                    active_loss = target_mask[..., ...].ne(0).view(-1) == 1
                    shift_labels = labels.contiguous()
                    scores = scores[..., :labels.shape[1]]
                    scores = scores.contiguous()

                    lm_logits=out.logits
                    shift_logits = lm_logits.contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss], scores.view(-1)[active_loss])
                    kl_loss = loss
                    out_aug = model(input_ids=source_ids_aug, attention_mask=source_mask, labels=labels, return_dict=True)
                    lm_logits_aug=out_aug.logits
                    shift_logits_aug = lm_logits_aug.contiguous()
                    loss_aug = loss_fct(shift_logits_aug.view(-1, shift_logits_aug.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss], scores.view(-1)[active_loss])
                    kl_loss = compute_kl_loss(shift_logits.view(-1, shift_logits.size(-1))[active_loss], shift_logits_aug.view(-1, shift_logits_aug.size(-1))[active_loss])
                    
                    loss = loss+loss_aug+args.k*kl_loss
                else:
                    loss = out.loss
                    kl_loss = loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                tr_kl_loss += args.k*kl_loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                kl_loss = round(tr_kl_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {} kl_loss {}".format(epoch, train_loss, kl_loss))

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval and epoch>8:
                # Calculate bleu
                this_bleu, dev_dataset = calculate_bleu(dev_filename, args, tokenizer, device, model, is_test=False, dev_dataset=dev_dataset, best_bleu=best_bleu)

                if this_bleu > best_bleu:
                    logger.info(" Achieve Best bleu:%s", this_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = this_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    best_model = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(best_model.state_dict(), output_model_file)

    if args.do_test:
        files = []
        if args.test_file is not None:
            for file_name in args.test_file.split(','):
                files.append(file_name)
        for idx, file in enumerate(files):
            calculate_bleu(file, args, tokenizer, device, model, file_postfix=str(idx), is_test=True)


def calculate_bleu(file_name, args, tokenizer, device, model, file_postfix=None, is_test=False, dev_dataset=None,
                   best_bleu=None):
    logger.info("BLEU file: {}".format(file_name))

    # whether append postfix to result file
    if file_postfix is not None:
        file_postfix = "_" + file_postfix
    else:
        file_postfix = ""

    if is_test:
        file_prefix = "test"
    else:
        file_prefix = "dev"

    # if dev dataset has been saved
    if (not is_test) and ('dev_bleu' in dev_dataset):
        eval_examples, eval_data = dev_dataset['dev_bleu']
    else:
        # read texts
        eval_examples = read_examples(file_name, args)

        # only use a part for dev
        if not is_test:
            eval_examples = random.sample(eval_examples, min(2000000, len(eval_examples)))

        # tokenize data
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')

        all_source_ids = eval_features['source_ids']
        all_source_mask = eval_features['source_mask']

        eval_data = TensorDataset(all_source_ids, all_source_mask)

        if not is_test:
            dev_dataset['dev_bleu'] = eval_examples, eval_data

    # get dataloader
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=5)

    model.eval()

    # generate texts by source
    generated_texts = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            generated_texts_ids = model.generate(input_ids=source_ids, attention_mask=source_mask,num_beams=args.beam_size,
                                                 max_length=args.max_target_length)

            for text_ids in generated_texts_ids:
                text = tokenizer.decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                generated_texts.append(text)

    # write to file
    predictions = []

    with open(os.path.join(args.output_dir, file_prefix + "{}.output".format(file_postfix)), 'w') as f, open(
            os.path.join(args.output_dir, file_prefix + "{}.gold".format(file_postfix)), 'w') as f1:

        for ref, gold in zip(generated_texts, eval_examples):
            predictions.append(str(gold.idx) + '\t' + ref)
            f.write(str(gold.idx) + '\t' + ref + '\n')
            f1.write(str(gold.idx) + '\t' + gold.target + '\n')

    # compute bleu
    (goldMap, predictionMap) = bleu.computeMaps(predictions,
                                                os.path.join(args.output_dir, file_prefix + "{}.gold".format(file_postfix)))
    this_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)

    if is_test:
        logger.info("  %s = %s " % ("bleu-4", str(this_bleu)))
    else:
        logger.info("  %s = %s \t Previous best bleu %s" % ("bleu-4", str(this_bleu), str(best_bleu)))

    logger.info("  " + "*" * 20)

    return this_bleu, dev_dataset


if __name__ == "__main__":
    my_args = read_arguments()

    # begin time
    begin_time = time.time()

    # logger for record
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    # write to file
    handler = logging.FileHandler('./log/log.txt')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # write to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # print args
    logger.info(my_args)

    main(my_args)

    logger.info("Finish training and take %s", get_elapse_time(begin_time))


