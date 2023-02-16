import json

import streamlit as st
from nnsplit import NNSplit

from models import get_answer


@st.cache_resource
def load_splitter():
    return NNSplit.load("en")


def _main():
    splitter = load_splitter()

    with open("context_embeddings.json") as f:
        all_contexts = json.load(f)

    st.header("⚡ 🧙 Potter Head")
    st.image("assets/potterhead.png")

    question = st.text_input("Hey! Ask anything about Harry Potter!")
    if question:
        with st.spinner("Searching..."):
            answer = get_answer(all_contexts, question, splitter, api_key=st.secrets["OPENAI_API_KEY"])

        if isinstance(answer, dict):
            st.error(answer["error"])
        else:
            st.markdown(f"""<span style="word-wrap:break-word;">{answer}</span>""", unsafe_allow_html=True)

    st.write("##")
    st.write("##")
    st.write("##")
    st.write("##")
    st.write("##")
    st.markdown(
        """
        <p style='display: block; text-align: center;'>Made by <a href="https://twitter.com/imgrohit">@imgrohit</a></p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style='display: block; text-align: center;'>Built with <a href="https://openai.com/api/">Open AI GPT3</a></p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    _main()
