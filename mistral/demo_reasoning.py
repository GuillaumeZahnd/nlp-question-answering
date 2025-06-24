import os
from mistralai import Mistral


if __name__ == "__main__":

    api_key = os.environ["MISTRAL_API_KEY"]
    model = "magistral-medium-2506"

    client = Mistral(api_key=api_key)

    message_role = "user"
    message_context = "A Coup represents operations short of full-scale war to change the composition of a target country’s government. A player attempting a Coup need not have any Influence in the target country or in an adjacent country to attempt the Coup. However, your opponent must have Influence markers in the target country for a Coup to be attempted. To resolve a Coup attempt, multiply the Stability Number of the target country by two. Then roll a die and add the Operations points on the card to it. If this modified die roll is greater than the doubled stability number, the Coup is successful, otherwise it fails. If the Coup is successful remove opposing Influence markers equal to the difference from the target country. If there are insufficient opposing Influence markers to remove, add friendly Influence markers to make up the difference. Move the marker on the Military Operations track up the number of spaces equal to the Operations value of the card played. Any Coup attempt in a Battleground country degrades the DEFCON status one level (towards Nuclear War). Cards that state a player may make a 'free Coup roll' in a particular region may ignore the geographic restrictions of the current DEFCON level. However, a 'free Coup roll' used against a Battleground country would still lower DEFCON."
    message_question = "The US player plays a 3 Operations card to conduct a coup attempt in Mexico. The US player has no Influence in Mexico; the USSR player has 2 Influence points. The US player rolls the die for a 4. After resolving the Coup, how much influence does each player have in Mexico?"
    message_content = "{} {}".format(message_context, message_question)

    ground_truth_answer = "After resolving the Coup, there is 1 (one) US influence and 0 (zero) USSR influence in Mexico."

    chat_response = client.chat.complete(
        model=model,
        messages=[{"role": message_role, "content": message_content}],
        prompt_mode="reasoning"
    )
    response = chat_response.choices[0].message.content

    print("Context: {}\n".format(message_context))
    print("Question: {}\n".format(message_question))
    print("Ground-truth answer: {}".format(ground_truth_answer))
    print("Response {}".format(response))
