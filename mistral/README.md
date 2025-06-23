# Getting started

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
