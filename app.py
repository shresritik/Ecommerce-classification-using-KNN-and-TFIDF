import streamlit as st
import pickle
import spacy
nlp = spacy.load("en_core_web_lg")

data = {
    "Household": 0,
    "Books": 1,
    "Electronics": 2,
    "Clothing & Accessories": 3
}


def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return [" ".join(filtered_tokens)]


st.title('Ecommerce Product Classifier')


# Create a text element and let the reader know the data is loading.
data_load_state = st.text_input("Enter Product Details")
with open('model/ecommerce.pkl', 'rb') as f:
    model = pickle.load(f)

    if (data_load_state != ""):
        text = preprocess(data_load_state)
        pred = model.predict(text)

        matching_keys = [key for key,
                         value in data.items() if value == pred[0]]
        st.text("Output: " + matching_keys[0])
