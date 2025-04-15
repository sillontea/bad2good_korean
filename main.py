import torch
import time
from models import StyleTransformer, Discriminator
from StyleTransformerB2G import StyleTransformerB2G
from HateClassifier import HateClassifier

class Application(StyleTransformerB2G, HateClassifier):
    def __init__(self):
        self.model = StyleTransformerB2G()
        self.cls = HateClassifier()
        
def main():    
    app = Application()

    while True:
        sentence = input('\n문장을 입력해주세요. (종료 시 q)\n') 
        
        if sentence == 'q':
            break

        try:
            # stage 1: filter
            filtted = app.cls.filter(sentence)

            # stage 2: delete or style transfer
            if filtted == 'extreme_hate':
                print(filtted)
                print('warning: 말 이쁘게 안 해?\n')
                out_sent = '바르고 고운말을 사용합시다.'
                print(out_sent)

            elif filtted == 'hate':
                out_sent = app.model.bad2good(sentence)
                print(filtted)
                print(out_sent)

            else:
                out_sent = sentence
                print(filtted)
                print(out_sent)

        except:
            pass



if __name__ == '__main__':
    main()
