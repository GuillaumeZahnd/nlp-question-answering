import torch
import numpy as np
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer


def question_answering_bert_finetuned_squad_wrapper(context: str, question: str) -> str:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_checkpoint = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    model = BertForQuestionAnswering.from_pretrained(model_checkpoint).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    
    batch_encoding_tmp = tokenizer.encode_plus(text=question, text_pair=context)
    input_ids = batch_encoding_tmp['input_ids']
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    segment_ids = get_segment_ids_from_input_ids(input_ids=input_ids, tokenizer=tokenizer)
    
    question_answering_model_output_tmp = model(
        torch.tensor([input_ids]).to(device),
        token_type_ids=torch.tensor([segment_ids]).to(device))    
        
    start_logits = question_answering_model_output_tmp['start_logits'][0].detach().cpu().numpy()
    end_logits = question_answering_model_output_tmp['end_logits'][0].detach().cpu().numpy()        
        
    answer_start = np.argmax(start_logits)
    answer_end = np.argmax(end_logits)
    answer = ' '.join(tokens[answer_start:answer_end+1])
    
    return answer

    
def get_segment_ids_from_input_ids(input_ids, tokenizer):
    # input:  '[CLS]', 'x', 'x', '[SEP]', 'x', 'x', 'x', 'x', 'x', '[SEP]'
    # output: [0,       0,   0,   0,       1,   1,   1,   1,   1,   1] 
    separator_index = input_ids.index(tokenizer.sep_token_id)
    length_of_first_segment = separator_index + 1
    length_of_second_segment = len(input_ids) - length_of_first_segment
    segment_ids = [0]*length_of_first_segment + [1]*length_of_second_segment    
    return segment_ids
