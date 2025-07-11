import os
import pickle
import spacy
from transformers import AutoTokenizer

from jax_mlm_helpers import build_vocabulary
from jax_mlm_helpers import apply_random_masking
from jax_mlm_helpers import pad_and_crop_to_maximum_length


def construct_mlm_dataset() -> None:

    # Read text file
    with open("local_datasets/wikipedia_man_o_war.txt", "r") as fid:
        text = fid.read()

    # Handle new lines
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")

    # Tokenize text into sentences
    nlp = spacy.load("en_core_web_sm")
    sentences = [s for s in nlp(text).sents]

    # Tokenize words into indices
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokens = []
    for s in sentences:
        tokens.append(tokenizer.tokenize(str(s)))

    # Create dictionaries
    dico_word2index, dico_index2word = build_vocabulary(texts=tokens)

    # Construct dataset
    dataset = []

    minimum_length = 5
    masking_probability = 0.15
    label_for_unmasked_values = -100
    maximum_sequence_length = 50
    mask_index = dico_word2index["[MASK]"]
    pad_index = dico_word2index["[PAD]"]
    padding_value_for_label = -100

    for t in tokens:
        if len(t) >= minimum_length:
            indices = [dico_word2index.get(w, "[UNK]") for w in t]

            # Apply random masling
            input_indices, mask, masked_indices, labels = apply_random_masking(
                input_indices=indices,
                index_for_masked_values=mask_index,
                label_for_unmasked_values=label_for_unmasked_values,
                masking_probability=masking_probability)

            # Pad and crop to maximum length
            input_indices = pad_and_crop_to_maximum_length(
                input_indices, padding_value=pad_index, maximum_sequence_length=maximum_sequence_length)
            mask = pad_and_crop_to_maximum_length(
                mask, padding_value=pad_index, maximum_sequence_length=maximum_sequence_length)
            masked_indices = pad_and_crop_to_maximum_length(
                masked_indices, padding_value=pad_index, maximum_sequence_length=maximum_sequence_length)
            labels = pad_and_crop_to_maximum_length(
                labels, padding_value=padding_value_for_label, maximum_sequence_length=maximum_sequence_length)

            data = dict()
            data["input_indices"] = input_indices
            data["mask"] = mask
            data["masked_indices"] = masked_indices
            data["labels"] = labels
            dataset.append(data)

    # Save
    with open(os.path.join("local_datasets", "wikipedia_man_o_war.pkl"), "wb") as fid:
        pickle.dump([dico_word2index, dico_index2word, dataset], fid)
