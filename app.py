import json

import streamlit as st
from nnsplit import NNSplit
from streamlit_chat import message

from models import get_answer

if "answer" not in st.session_state:
    st.session_state["answer"] = []

if "question" not in st.session_state:
    st.session_state["question"] = []


@st.cache_resource
def load_splitter():
    return NNSplit.load("en")


def _main():
    splitter = load_splitter()

    with open("context_embeddings.json") as f:
        all_contexts = json.load(f)

    st.header("⚡ 🧙 Potter Head")
    st.image("assets/potterhead.png")

    st.markdown(
        """
        <p style='display: block; text-align: left;'>Made by <a href="https://twitter.com/imgrohit">@imgrohit</a>, Built with <a href="https://openai.com/api/">Open AI GPT3</a></p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style='display: block; text-align: left;'></p>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_input("Hey! Ask anything about Harry Potter!")
    if question:
        with st.spinner("Searching..."):
            answer = get_answer(
                all_contexts,
                question,
                splitter,
                prev_questions=st.session_state["question"],
                prev_answers=st.session_state["answer"],
                api_key=st.secrets["OPENAI_API_KEY"],
            )
            # answer = 'this is someting'

        if isinstance(answer, dict):
            st.error(answer["error"])
        else:
            st.session_state["question"].append(question)
            st.session_state["answer"].append(answer)

        for i in range(len(st.session_state["question"]) - 1, -1, -1):
            message(st.session_state["question"][i], is_user=True, key=f"{i}_ques")
            message(st.session_state["answer"][i], is_user=False, key=f"{i}_ans")
            # st.markdown(f"""<span style="word-wrap:break-word;">{answer}</span>""", unsafe_allow_html=True)


if __name__ == "__main__":
    _main()
