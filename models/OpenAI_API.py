# Licensed under the MIT license.

import os
import os
import time
from tqdm import tqdm
import concurrent.futures
# from openai import AzureOpenAI

# my imports
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv
load_dotenv()

# changed to azure version because other wasn't working for me
client = ChatCompletionsClient(
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["AZURE_OPENAI_KEY"]),
)

max_threads = 32


def load_OpenAI_model(model):
    return None, model


def generate_with_OpenAI_model(
    prompt,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
):
    messages = [SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=prompt),]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "seed": 1,
    }

    ans, timeout = "", 5
    while not ans:
        try:
            time.sleep(timeout)
            # completion = client.chat.completions.create(messages=messages, **parameters)
            completion = client.complete(
                    messages=messages,
                    **parameters
                )
            ans = completion.choices[0].message.content

        except Exception as e:
            print(e)
        if not ans:
            timeout = timeout * 2
            if timeout > 120:
                timeout = 1
            try:
                print(f"Will retry after {timeout} seconds ...")
            except:
                pass
    return ans


def generate_n_with_OpenAI_model(
    prompt,
    n=1,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
    max_threads=3,
    disable_tqdm=True,
):
    preds = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(generate_with_OpenAI_model, prompt, model_ckpt, max_tokens, temperature, top_k, top_p, stop)
            for _ in range(n)
        ]
        for i, future in tqdm(
            enumerate(concurrent.futures.as_completed(futures)),
            total=len(futures),
            desc="running evaluate",
            disable=disable_tqdm,
        ):
            ans = future.result()
            preds.append(ans)
    return preds


if __name__ == "__main__":
    print(generate_with_OpenAI_model("What is the meaning of life?", model_ckpt="gpt-35-turbo-16k"))
    print(generate_n_with_OpenAI_model("What is the meaning of life?", n=5, model_ckpt="gpt-35-turbo-16k", temperature=0.8, top_p=0.95, stop=["\n"]))