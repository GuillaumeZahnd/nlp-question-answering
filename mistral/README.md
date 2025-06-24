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

## `demo_rag.ipynb`

Context:

>  It fell to one of Ivan's successors, Ivan the Terrible, to seize the first two of these and incorporate them in Muscovy's rapidly expanding empire. Thirsting for revenge, his troops stormed the fortress of Kazan on the upper Volga in 1553, slaughtering the defenders just as the Mongols had done when they laid waste Russia's great cities. Two years later the Khanate of Astrakhan, where the Volga flows into the Caspian, met with a similar fate. Only the Crimea, the last remaining Tartar redoubt, still held out, and then merely because it enjoyed the protection of the Ottoman sultans, who regarded it as a valuable bulwark against the Russians. The invasions of Napoleon and Hitler, though unsuccessful, merely reinforced these fears. Only now do the Russian people show signs of shaking off this unhappy legacy. Those ferocious little horsemen whom Genghis Khan let loose upon the world have much to answer for, more than four centuries after their power was finally broken and they themselves sank back into the obscurity from which they had come. The man to whom the Russians owe their freedom from Mongol oppression was Ivan III, known also as Ivan the Great, then Grand Prince of Moscow.

Query:

> What did Ivan the Terrible achieve?

Response:

> Ivan the Terrible, one of Ivan's successors, achieved the seizure and incorporation of two significant territories into Muscovy's expanding empire. Specifically, he led his troops to storm the fortress of Kazan on the upper Volga in 1553, resulting in the slaughter of the defenders. Two years later, in 1555, the Khanate of Astrakhan, located where the Volga flows into the Caspian, met a similar fate.
