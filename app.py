import streamlit as st
import requests

st.set_page_config(page_title="HBO Document Q&A", page_icon="📺")
st.title("📺 HBO Document Q&A Assistant")
st.write("Ask a question about the HBO retrospective document.")

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question}
        )
        data = response.json()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Sources used")
    for i, source in enumerate(data["sources"]):
        with st.expander(f"Source {i+1} (relevance: {source['score']:.2f})"):
            st.write(source["text"])