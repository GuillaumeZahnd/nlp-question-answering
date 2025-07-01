conll2003 dataset
https://arxiv.org/pdf/cs/0306050v1

Hugging Face datasets.Dataset object, saved to disk using datasets.save_to_disk().

To load the datasets:

```sh
import datasets
dataset_train = datasets.load_from_disk("train.hf")
dataset_test = datasets.load_from_disk("test.hf")
``
