from nltk.tokenize import word_tokenize
from konlpy.tag import Mecab

from transformers import pipeline
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForMaskedLM
from nltk.translate.bleu_score import sentence_bleu

import torch
import os
import csv



# import fasttext
# import kenlm
# import pkg_resources
import math
import numpy as np
import pandas as pd

class Evaluator(object):
    def __init__(self):
        root = os.getcwd()
        
        # hate_ref file load_all(clean+hate)
        # index 0: hate, index 1: clean
        self.hate_ref = []
        
        # hate data
        with open(root+'/evaluator/hate_refs_0.csv' , 'r') as f:
            csv_f = csv.reader(f)
            self.hate_ref.append([line for line in csv_f])
        
        #clean data
        with open(root+'/evaluator/hate_refs_1.csv' , 'r') as f:
            csv_f = csv.reader(f)
            self.hate_ref.append([line for line in csv_f])
        
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        self.mecab = Mecab()

        # cls model 불러오기
        self.classifier_hate = pipeline('text-classification', model='beomi/beep-KcELECTRA-base-hate') #, device=0)
        
        self.LABEL_DIC = {
            'none': 0,
            'offensive': 1,
            'hate': 2,
        }
        
        # ppl용 MaskedLM
        self.ppl_hate = AutoModelForMaskedLM.from_pretrained("beomi/beep-KcELECTRA-base-hate")
        
  
    # 1. style
    def style_check(self, text_transferred, style_origin):
        x = self.classifier_hate(text_transferred)[0]['label']
        
        if text_transferred == '':
            return False
        
        style_transffered = [0 if self.LABEL_DIC[x] > 0 else 1][0] # 0:hate 1:clean || positive: 1 negative: 0
        
        # style이 다르면 True, 같으면 False
        return (style_transffered != style_origin)
    
    # style check batch
    def hate_acc_b(self, texts, styles_origin):
        assert len(texts) == len(styles_origin), 'Size of inputs does not match!'
        count = 0
        for text, style in zip(texts, styles_origin):
            if self.style_check(text, style):
                count += 1
        return count / len(texts)

    # original이 neg(0)일 때 
    def hate_acc_0(self, texts):
        styles_origin = [0] * len(texts)
        return self.hate_acc_b(texts, styles_origin)

    # original이 pos(1)일 때 
    def hate_acc_1(self, texts):
        styles_origin = [1] * len(texts)
        return self.hate_acc_b(texts, styles_origin)


    
    # 2. contents
    def nltk_bleu(self, texts_origin, text_transferred):
        texts_origin = [self.mecab.morphs(text_origin.lower().strip()) for text_origin in texts_origin]
        text_transferred = self.mecab.morphs(text_transferred.lower().strip())

        return sentence_bleu(texts_origin, text_transferred) * 100

    # not_use
    def self_bleu_b(self, texts_origin, texts_transfered):
        assert len(texts_origin) == len(texts_transfered), 'Size of inputs does not match!'
        sum = 0
        n = len(texts_origin)
        for x, y in zip(texts_origin, texts_transfered):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def hate_ref_bleu_0(self, texts_neg2pos):
        assert len(texts_neg2pos) == 10, 'Size of input differs from human reference file(10)!'
        sum = 0
        n = 10
        for x, y in zip(self.hate_ref[0], texts_neg2pos):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def hate_ref_bleu_1(self, texts_pos2neg):
        assert len(texts_pos2neg) == 10, 'Size of input differs from human reference file(10)!'
        sum = 0
        n = 10
        for x, y in zip(self.hate_ref[1], texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n
    
    
    # 3. ppl
    def bert_ppl(self, sentence):
        tokenizer = self.tokenizer
        model = self.ppl_hate

        tensor_input = tokenizer.encode(sentence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)

        with torch.inference_mode():
            loss = model(masked_input, labels=labels).loss

        return np.exp(loss.item())
    
    



