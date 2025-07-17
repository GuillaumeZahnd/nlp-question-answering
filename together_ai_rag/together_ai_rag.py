import os
from together import Together


if __name__ == "__main__":

    api_key = os.environ.get("TOGETHER_API_KEY")

    client = Together(api_key=api_key)

    content = "Why does throwing a projectile at a 45° angle produce the greatest possible distance?"

    message = {"role": "user", "content": content}

    # Free models
    model_deepseek = "deepseek-ai/DeepSeek-R1-0528"
    model_llama = "meta-llama/Llama-Vision-Free"

    response_deepseek = client.chat.completions.create(model=model_deepseek, messages=[message])
    response_llama = client.chat.completions.create(model=model_llama, messages=[message])

    print("-"*64)

    print("[Question]")
    print(content)
    print("-"*64)

    print("[DeepSeek]")
    print(response_deepseek.choices[0].message.content)
    print("-"*64)

    print("[Llama]")
    print(response_llama.choices[0].message.content)
    print("-"*64)
    
    content += " Answer this question using only internal knowledge, do not search for additional external information."

    message = {"role": "user", "content": content}

    # Free models
    model_deepseek = "deepseek-ai/DeepSeek-R1-0528"
    model_llama = "meta-llama/Llama-Vision-Free"

    response_deepseek = client.chat.completions.create(model=model_deepseek, messages=[message])
    response_llama = client.chat.completions.create(model=model_llama, messages=[message])

    print("-"*64)

    print("[Question]")
    print(content)
    print("-"*64)

    print("[DeepSeek]")
    print(response_deepseek.choices[0].message.content)
    print("-"*64)

    print("[Llama]")
    print(response_llama.choices[0].message.content)
    print("-"*64)    
