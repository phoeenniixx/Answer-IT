import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("Answer IT")
st.sidebar.title("Text Section")
model_id = "google/flan-t5-large"
text = st.sidebar.text_area("Enter the text")
placeholder = st.empty()

tokenizer = AutoTokenizer.from_pretrained(model_id)
placeholder.text("downloading the models")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

query = st.text_input("Question: ")
process_text = st.button("Ask Question")
if process_text:

    placeholder.text("Text Splitter...Started...✅✅✅")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_text(text)

    prompt = f"""{docs}
    QUERY: {query}
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))

