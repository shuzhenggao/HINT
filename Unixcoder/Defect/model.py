# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class SCELoss(torch.nn.Module):
    def __init__(self, config, num_labels, alpha=1, beta=1):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.config = config
        self.num_labels = num_labels
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, pred, labels, score=None):
        # CCE
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_labels).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        if score is None:
            loss = (self.alpha * ce).mean() + self.beta * rce.mean() 
        else:
            loss = (self.alpha * ce).mean() + self.beta * rce.mean() 
        return loss

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss)
    return loss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.num_labels = 2
        if 'pl_ours' in self.args.mode:
            self.loss_fct = SCELoss(self.config, self.num_labels, alpha=1, beta=1)
            #self.loss_fct = CrossEntropyLoss()
        else:
            self.loss_fct = CrossEntropyLoss()
    
        
    def forward(self, input_ids=None,labels=None,score=None,input_aug_ids=None): 
        logits=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        if labels is not None:
            if 'pl_ours' in self.args.mode:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1), score.view(-1))
                logits_aug=self.encoder(input_aug_ids,attention_mask=input_aug_ids.ne(1))[0]
                loss_aug = self.loss_fct(logits_aug.view(-1, self.num_labels), labels.view(-1), score.view(-1))
                kl_loss = compute_kl_loss(logits, logits_aug)
                loss = loss+loss_aug+self.args.k*kl_loss
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits
    
        
        
        
 
