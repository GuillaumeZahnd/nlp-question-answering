# Getting started

Depending on your operating system (``<os>``, linux or windows), rename the file ``Pipfile_<os>`` into ``Pipfile``.

```
mkdir .venv
python -m pip install --upgrade setuptools pip
pipenv install -d --python 3.10
```

## Jupyter notebook using pipenv

0. Create a pipenv shell:

```sh
pipenv shell
```

1. Install the pipenv kernelspec for jupyter:

```sh
python -m ipykernel install --user --name=`basename $VIRTUAL_ENV`
```

2. Launch the jupyter notebook:

```sh
jupyter notebook
```

3. From the notebook, select the `.venv` kernel.

## spaCy

```
pipenv shell
python -m spacy download en_core_web_sm
```
