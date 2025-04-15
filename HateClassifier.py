from transformers import pipeline

class HateClassifier:
    def __init__(self):       
        print('잠시만 기다려 주세요. 모델을 불러옵니다...\n')
        self.pipe = pipeline('text-classification', model='beomi/beep-KcELECTRA-base-hate') # ,device=0
        
        print('done! \n')
        self.LABEL_DIC = {
            'none': 0,
            'offensive': 1,
            'hate': 2,
        }
        

    def filter(self, sent):   
        # 문서 단위 들어올 때 고려 필요
        # results = df['comments'].map(lambda x: pipe(x)[0])

        # labels, scores = [], []

        # for line in results:
        #     labels.append(line['label'])
        #     scores.append(line['score'])    
        
        out = self.pipe(sent)[0]
        label, score = out['label'], out['score']

        label = self.LABEL_DIC[label]

        if label != 0 and score >= 0.98:
            return 'extreme_hate'
        elif label != 0 and score < 0.98:
            return 'hate'
        else: 
            return 'clean'
    

