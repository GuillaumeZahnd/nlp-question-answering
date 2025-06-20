import numpy as np


def softmax(x, axis=0):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    

def get_model_and_tokenizer(model_checkpoint, device):

    if model_checkpoint == 'bert-large-uncased-whole-word-masking-finetuned-squad':
        from transformers import BertForQuestionAnswering
        from transformers import BertTokenizer
        model = BertForQuestionAnswering.from_pretrained(model_checkpoint)
        tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    
    elif model_checkpoint == 'bert-base-uncased':
        from transformers import BertForQuestionAnswering
        from transformers import BertTokenizer
        model = BertForQuestionAnswering.from_pretrained(model_checkpoint)
        tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
        
    elif model_checkpoint == 'distilbert-base-uncased':
        from transformers import AutoModelForQuestionAnswering
        from transformers import AutoTokenizer
        model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    else:
        raise ValueError('Model not supported: {}'.format(model_checkpoint))
    
    return model.to(device), tokenizer    
