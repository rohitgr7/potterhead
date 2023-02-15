import json

import streamlit as st

from models import get_answer


def _main():
    with open("context_embeddings.json") as f:
        all_contexts = json.load(f)

    st.header("⚡ 🧙 Potter Head")
    st.image("assets/potterhead.png")

    question = st.text_input("Hey! Ask anything about Harry Potter!")
    if question:
        with st.spinner("Searching..."):
            answer = get_answer(all_contexts, question, api_key=st.secrets["OPENAI_API_KEY"])
        st.markdown(f"""<span style="word-wrap:break-word;">{answer}</span>""", unsafe_allow_html=True)


if __name__ == "__main__":
    _main()
