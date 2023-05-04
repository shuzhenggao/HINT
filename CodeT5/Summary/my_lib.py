import json
import time
import torch
import random
from torch.utils.data import Dataset
#from openprompt.data_utils import InputExample


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 original_source,
                 original_target,
                 score
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.origin_source = original_source
        self.origin_target = original_target
        self.code=''
        self.score = score


def read_examples(filename, args):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
                
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())

            if 'score' not in js:
                score = 0
            else:
                score=js['score']

            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                    original_source=js['code_tokens'],
                    original_target=js['docstring_tokens'],
                    score=score,
                )
            )

    return examples


class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    # collect texts
    codes = []
    target_nl = []
    scores = []
    for example_id, example in enumerate(examples):
        codes.append(example.source)
        scores.append([example.score]*args.max_source_length)

        if stage == "test":
            target_nl.append(example.target)
            #target_nl.append("None")
        else:
            target_nl.append(example.target)

    # begin tokenizing
    encoded_codes = tokenizer(
        codes, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    encoded_targets = tokenizer(
        target_nl, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    return {'source_ids':encoded_codes['input_ids'], 'target_ids':encoded_targets['input_ids'],
            'source_mask':encoded_codes['attention_mask'], 'target_mask':encoded_targets['attention_mask'], 'all_scores':torch.Tensor(scores)}



class TextDataset(Dataset):
    def __init__(self, tokenizer, args, model_config, features, mode,idx=None):
        self.examples = [{} for _ in range(len(features['source_ids']))] 
        for key in features:
            for idx in range(len(features[key])):
                self.examples[idx][key] = features[key][idx]
        self.tokenizer = tokenizer
        self.mode = mode
        self.args = args
        self.model_config = model_config

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_ids = self.examples[i]['source_ids'].clone().detach()
        #print(self.examples[i]['all_scores'][0])
        #if self.mode and self.examples[i]['all_scores'][0]<1:
        '''if True:
            if random.random()<(1-self.examples[i]['all_scores'][0]):
                length = sum([obj != self.tokenizer.pad_token_id for obj in input_ids]) 
                pos = random.sample(range(length),length//20)
                for idx in pos:
                    #sub_idx = random.randint(15, self.model_config.vocab_size-1)
                    input_ids[idx] = self.tokenizer.mask_token_id'''
        return self.examples[i]['source_ids'], self.examples[i]['source_mask'], self.examples[i]['target_ids'], self.examples[i]['target_mask'], self.examples[i]['all_scores'], input_ids
