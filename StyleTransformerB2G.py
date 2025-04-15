import time

import torch
from transformers import pipeline
from nltk.tokenize import word_tokenize
# from konlpy.tag import Mecab

from config import Config
from utils import *
from models import StyleTransformer, Discriminator

class StyleTransformerB2G:
    def __init__(self):     
        # mecab = Mecab()
        # self.tokenizer = mecab.morphs

        self.config = Config() # config에 pretrained_root, f_path, d_path 추가    
        self.load_vocab()
        self.load_model()
    
    def load_vocab(self):
        path = self.config.vocab_path+'vocab3.txt'
        self.vocab = torch.load(path)
        print('Vocab size:', len(self.vocab))

    def load_model(self):
        config = self.config
        vocab = self.vocab

        # 1. 모델 불러오기
        self.model_F = StyleTransformer(config, vocab).to(config.device)
        self.model_D = Discriminator(config, vocab).to(config.device)

        print('discriminator_method : ', config.discriminator_method)

        # 2. 가중치 불러오기
        # 스타일변환기(f), 스타일판별기(d)
        f_path = config.pretrained_root+'125_F.pth' 
        d_path = config.pretrained_root+'125_D.pth'

        weights_F = torch.load(f_path, map_location=config.device)
        weights_D = torch.load(d_path,  map_location=config.device)

        self.model_F.load_state_dict(weights_F)
        self.model_D.load_state_dict(weights_D)
        
        print('\n<All keys matched successfully>')

    def preprocess_text(self, sent):
        config = self.config
        vocab = self.vocab
        
        tokens =  word_tokenize(sent) # 토큰화 self.tokenizer(sent) #
        tokens_idx = [vocab.stoi[token] for token in tokens] # 단어 -> 인덱스
        tensor = torch.tensor(tokens_idx) # to_tensor
        
        eos_idx = vocab.stoi['<eos>']
        max_length = config.max_length
        pad_idx = vocab.stoi['<pad>']
        
        # max_length 보다 짧은 문장 pad 추가
        diff = max_length - len(tensor)
        pad = torch.tensor([pad_idx] * diff)
        inp_tokens = torch.cat((tensor, pad), 0)
        inp_tokens = inp_tokens.view(1,16)
        
        # 모델 입력 모양으로 변경
        batch = torch.cat((inp_tokens, inp_tokens), 0)
        
        return batch

    def inference_sample(self, inp_tokens, raw_style, temperature=0.5):
        vocab = self.vocab
        vocab_size = len(vocab)
        eos_idx = vocab.stoi['<eos>']
        
        model_F = self.model_F
        model_D = self.model_D
        
        gold_text = []
        raw_output = []
        rev_output = []
        
        model_F.eval()
        
        inp_lengths = get_lengths(inp_tokens, eos_idx)
        raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
        rev_styles = 1 - raw_styles

        with torch.no_grad():
            raw_log_probs = model_F(
                inp_tokens,
                None,
                inp_lengths,
                raw_styles,
                generate=True,
                differentiable_decode=False,
                temperature=temperature,
            )

        with torch.no_grad():
            rev_log_probs = model_F(
                inp_tokens, 
                None,
                inp_lengths,
                rev_styles,
                generate=True,
                differentiable_decode=True,
                temperature=temperature,
            )
        
        gold_text += tensor2text(vocab, inp_tokens.cpu())
        raw_output += tensor2text(vocab, raw_log_probs.argmax(-1).cpu())
        rev_output += tensor2text(vocab, rev_log_probs.argmax(-1).cpu())

        return gold_text, raw_output, rev_output    


    def bad2good(self, bad_sent):
        bad = bad_sent
        batch = self.preprocess_text(bad)
        gold, raw, rev = self.inference_sample(batch, raw_style=0)
        
        good = rev[0]
        
        print('\nbad_sentence: ', bad)
        print('good_sentence: ', good)   
        
        return gold[0], raw[0], rev[0]


    def _check_inferred(self, gold, raw, rev):
        output = zip(gold, raw, rev)
        
        print('original  --> raw_sent --> reverse\n')
        for before, reduct, after in output:
            print(before, ' --> ', reduct, ' --> ', after)


 
