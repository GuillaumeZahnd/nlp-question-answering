# Getting started

Depending on your operating system (``<os>``, linux or windows), rename the file ``Pipfile_<os>`` into ``Pipfile``.

```
mkdir .venv
python -m pip install --upgrade setuptools pip
pipenv install -d --python 3.10
```

## Jupyter notebook using pipenv

1. Create a pipenv shell:

```sh
pipenv shell
```

2. Install the pipenv kernelspec for jupyter:

```sh
python -m ipykernel install --user --name=`basename $VIRTUAL_ENV`
```

3. Launch the jupyter notebook:

```sh
jupyter notebook
```

4. From the notebook, select the `.venv` kernel.

## spaCy

```
pipenv shell
python -m spacy download en_core_web_sm
```

## API keys

API keys are required to utilize the API from [Mistral AI](https://console.mistral.ai/home) and [Together AI](https://api.together.ai/). The example below details how to set-up an API key for ``Mistral AI``.

1. Set up a Mistral account and create a new API key (see [Mistral quickstart](https://docs.mistral.ai/getting-started/quickstart/)).
2. Save the API key in a password manager.
3. Add the API key as an environment variable by adding the following line in the `~/.bashrc` file:

```sh
export MISTRAL_API_KEY="<the API key>" 
```
4. Run the `~./bashrc` file so the changes take effect:

```sh
source ~/.bashrc
```
5. Verify that the API is known as an environment variable:

```sh
echo $MISTRAL_API_KEY
```

6. Minimal Pythonic working example to instanciate a client with an API key:

```sh
import os
from mistralai import Mistral
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)
```
