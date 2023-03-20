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


def get_standalone_question(question, prev_questions, prev_answers, api_key):
    history = [(ques, ans) for ques, ans in zip(prev_questions, prev_answers)]
    promp = f"""
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone
    question.

    Chat History:
    {history}
    Follow Up Input: {question}
    Standalone question
    """

    headers = {"Content-type": "application/json", "Authorization": f"Bearer {api_key}"}
    json_data = {
        "model": "text-davinci-003",  # text-davinci-003
        "prompt": promp,
        "temperature": 0.2,
        "max_tokens": 128,
    }
    response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=json_data)
    return response.json()["choices"][0]["text"].strip()


def generate_answer(context, question, api_key):
    prompt = f"""
    As an AI assistant, your role is to provide accurate responses to questions. You will be presented with excerpts
    from a lengthy movie plot and a corresponding query. Your response should be conversational and based on the given
    context. If you are unable to find the answer within the context, simply state, "Hmm, I'm not sure." Please refrain
    from fabricating an answer if one cannot be found. If the question is unrelated to the context, kindly inform the
    user that your expertise is limited to answering context-related questions.

    QUESTION:

    {question}

    PASSAGE:
    {context}

    ANSWER:
    """

    headers = {"Content-type": "application/json", "Authorization": f"Bearer {api_key}"}
    json_data = {
        "model": "text-davinci-003",  # text-davinci-003
        "prompt": prompt,
        "temperature": 0.2,
        "max_tokens": 128,
    }
    response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=json_data)
    return response.json()["choices"][0]["text"].strip()


def get_answer(all_contexts, question, splitter, prev_questions, prev_answers, api_key):
    standalone_question = get_standalone_question(question, prev_questions, prev_answers, api_key)
    ques_embed = get_embeddings(standalone_question, api_key)

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
    answer = generate_answer(search_context, standalone_question, api_key)

    if answer.lower() == "i don't know.":
        return "That's an interesting question, but I'm not sure it's quite relevant."

    splits = splitter.split([answer])[0]
    data = [str(x).strip() for x in splits]

    if not data[-1].endswith("."):
        data = data[:-1]

    answer = " ".join(data)
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
        for i in tqdm(range(0, len(data), step), leave=False, total=len(data) // step):
            context = "".join(data[i : i + step])
            embedding = get_embeddings(context)
            all_context.append({"context": context, "embedding": embedding})

    with open("context_embeddings.json", "w") as f:
        json.dump(all_context, f)
