import numpy as np
import tensorflow as tf
import torch


def softmax_numpy(x, axis=-1):
    x_exp = np.exp(x) 
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)
    
def softmax_pytorch(x, dim=-1):
    x_exp = torch.exp(x) 
    return np.exp(x) / x_exp.sum(np.exp(x), dim=dim, keepdims=True)
    
def softmax_tensorflow(x, axis=1):
    x_exp = tf.exp(x)
    return np.exp(x) / tf.math.reduce_sum(x_exp, axis=axis, keepdims=True)        
    

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute the scaled dot product
    
    Args:
        q: Queries sequence, in the shape (sequence_length, hidden_dimensionality_for_queries).
        k: Keys sequence, in the shape (sequence_length, hidden_dimensionality_for_keys).        
        v: Values sequence, in the shape (sequence_length, hidden_dimensionality_for_values).
        mask: optional masking of specific entries in the attention matrix, in the shape (sequence_length, sequence_length).
        
    Returns:
        Attention weights, in the shape (sequence_length, sequence_length).
        Weighted sum of values, in the shape (sequence_length, sequence_length).
    """

    sequence_length_q, hidden_dimensionality_for_queries = q.shape
    sequence_length_k, hidden_dimensionality_for_keys = k.shape
    sequence_length_v, hidden_dimensionality_for_values = v.shape

    assert sequence_length_q == sequence_length_k
    assert sequence_length_q == sequence_length_v
    assert hidden_dimensionality_for_queries == hidden_dimensionality_for_keys
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (1. - mask) * -1e9 
    
    attention_weights = softmax_tensorflow(scaled_attention_logits)
    
    #attention_weights = tf.keras.activations.softmax(scaled_attention_logits)

    #assert tf.math.reduce_sum(attention_weights, axis=1)[0] == 1, "Each row in weights must sum to 1"
    
    weighted_sum_of_values = tf.matmul(attention_weights, v)
    
    assert weighted_sum_of_values.shape == v.shape
    
    return attention_weights, weighted_sum_of_values
    


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
