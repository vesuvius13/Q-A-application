import streamlit as st
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def qa_model():
    model = pipeline("question-answering")
    return model

qa = qa_model()
st.title("Ask Questions about your Text")
st.header("You can paste any article in the text area below and then use the input area to ask the question.")
st.subheader("This application is using HuggingFace's transformers pre-trained model to detect the context of the question and find the answer accordingly.")
sentence = st.text_area('Paste your copied data here...', height=30)
question = st.text_input('Your question...')
button = st.button("Get Answer")
with st.spinner("Loading Answer..."):
    if button and sentence:
        answers = qa(question=question, context=sentence)
        st.write(answers['answer'])
