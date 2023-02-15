import json
import os
import re
from operator import itemgetter
from pathlib import Path

import numpy as np
import requests
from nnsplit import NNSplit
from tqdm.auto import tqdm


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def remove_space(text):
    for space in spaces:
        text = text.replace(space, " ")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = remove_space(text)
    return text


def get_embeddings(text, api_key=None):
    if api_key is None:
        api_key = os.environ["OPENAI_API_KEY"]

    HEADERS = {"Content-type": "application/json", "Authorization": f"Bearer {api_key}"}

    json_data = {
        "input": text,
        "model": "text-embedding-ada-002",
    }
    response = requests.post("https://api.openai.com/v1/embeddings", headers=HEADERS, json=json_data)

    if response.status_code == 429:
        return {"error": "Limit exceeded! Please try after some time."}

    return response.json()["data"][0]["embedding"]


def generate_answer(context, question, api_key):
    HEADERS = {"Content-type": "application/json", "Authorization": f"Bearer {api_key}"}

    prompt = f"""
    Write a descriptive answer in past sense to the following question given the following passage. If you can't find an answer, just say I don't know:

    PASSAGE:
    {context}

    QUESTION:
    {question}

    ANSWER:

    """

    json_data = {
        "model": "text-davinci-003",  # text-davinci-003
        "prompt": prompt,
        "temperature": 1,
        "max_tokens": 128,
    }
    response = requests.post("https://api.openai.com/v1/completions", headers=HEADERS, json=json_data)
    return response.json()["choices"][0]["text"].strip()


def get_answer(all_contexts, ques, api_key):
    ques_embed = get_embeddings(ques, api_key)

    if isinstance(ques_embed, dict) and "error" in ques_embed:
        return ques_embed

    sim = []

    for i, x in enumerate(all_contexts):
        cosine_sim = cosine_similarity(x["embedding"], ques_embed)

        if cosine_sim > 0.5:
            sim.append((i, cosine_sim))

    if not sim:
        return "That's an interesting question, but I'm not sure it's quite relevant."

    sim = sorted(sim, key=lambda x: -x[1])
    ids = [i for i, _ in sim[:2]]

    contexts = list(itemgetter(*ids)(all_contexts))
    search_context = "".join(x["context"] for x in contexts)
    answer = generate_answer(search_context, ques, api_key)

    if answer.lower() == "i don't know.":
        return "That's an interesting question, but I'm not sure it's quite relevant."

    return answer


if __name__ == "__main__":
    spaces = [
        "\u200b",
        "\u200e",
        "\u202a",
        "\u202c",
        "\ufeff",
        "\uf0d8",
        "\u2061",
        "\x10",
        "\x7f",
        "\x9d",
        "\xad",
        "\xa0",
        "\u202f",
    ]
    splitter = NNSplit.load("en")

    all_context = []
    for fpath in tqdm(list(Path("book_summary").iterdir())):
        with open(fpath) as f:
            data = f.read()

        data = clean_text(data)

        splits = splitter.split([data])[0]
        data = [str(x) for x in splits]

        step = 10

        for i in tqdm(range(0, len(data), 15), leave=False, total=len(data) // 15):
            context = "".join(data[i : i + 15])
            embedding = get_embeddings(context)
            all_context.append({"context": context, "embedding": embedding})

    with open("context_embeddings.json", "w") as f:
        json.dump(all_context, f)
