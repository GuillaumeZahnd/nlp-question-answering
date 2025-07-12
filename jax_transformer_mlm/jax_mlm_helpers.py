import numpy as np
import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import random
from typing import List
import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")


def build_vocabulary(texts, max_vocab=10000, min_freq=1):

    word_counter = Counter()
    for doc in texts:
        for word in doc:
            word_counter[word.lower()] += 1

    dico_word2index = {}
    dico_index2word = {}

    special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
    for token in special_tokens:
        index = len(dico_word2index)
        dico_word2index[token] = index
        dico_index2word[index] = token

    for word, count in word_counter.most_common():
        if count < min_freq: break
        if len(dico_word2index) >= max_vocab: break
        index = len(dico_word2index)
        dico_word2index[word] = index
        dico_index2word[index] = word

    return dico_word2index, dico_index2word


def apply_word_tokenization(text) -> List[str]:
    nlp = spacy.blank("en")
    doc = nlp(text)
    word_tokens = [str(w).lower() for w in doc]
    return word_tokens


def apply_random_masking(
    input_indices: List,
    index_for_masked_values: int,
    label_for_unmasked_values: int=-100,
    masking_probability: int=0.15) -> tuple[Array, Array, Array, Array]:

    """
    XXX

    Args:
        input_indices:
        index_for_masked_values:
        label_for_unmasked_values:
        masking_probability:

    Returns:
        X
        X
        X
        X

    """

    sequence_length_before_padding = len(input_indices)

    main_rng = random.PRNGKey(421)
    _, sub_rng = random.split(main_rng)
    random_masking_probabilities = random.uniform(sub_rng, shape=(sequence_length_before_padding))

    input_indices = jnp.array(np.array(input_indices), dtype="int32")

    mask = jnp.zeros((sequence_length_before_padding), dtype="int32")
    mask = mask.at[random_masking_probabilities<masking_probability].set(1)

    masked_indices = input_indices.at[mask==1].set(index_for_masked_values)

    labels = input_indices.at[mask==0].set(label_for_unmasked_values)

    return input_indices, mask, masked_indices, labels


def pad_and_crop_to_maximum_length(x: ArrayLike, padding_value: int, maximum_sequence_length: int=20) -> Array:

    """
    XXX

    Args:
        x:
        padding_value:
        maximum_sequence_length:

    Returns:
        X
    """

    pad_width_after = max(0, maximum_sequence_length - len(x))
    x = jnp.pad(x, constant_values=padding_value, pad_width=(0, pad_width_after))
    x = x[:maximum_sequence_length]
    return x
