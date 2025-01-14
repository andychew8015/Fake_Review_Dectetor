import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Create the app title
st.title('Fake Review Detector')

# Load fine-tuned model, tokenizer and create a pipeline
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('../DistilBert_model')
clf = pipeline('text-classification', model=model, tokenizer=tokenizer, truncation=True, padding=True, max_length=75)

# Create button for input type
input_type = st.radio('Choose an input type', ['Upload a file as input', 'Enter text as input'])

if input_type == 'Upload a file as input':
    st.write('You can upload a file with .csv or .xlsx format')
    
    # Create a file uploader
    file = st.file_uploader('Upload file')
        
    # If file is uploaded, read the file using pandas
    # display error message if file type is invalid
    if file is not None:
        if file.type == 'text/csv':
            df = pd.read_csv(file)
        elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            df = pd.read_excel(file)
        else:
            st.error('Invalid file type. Please upload a file with .csv or .xlsx format.')
            st.stop()

        # Create a selectbox for column name
        col = st.selectbox('Choose a column name that contains raw texts data:', df.columns, index=None)

        # Convert selected column to list and display the first 5 rows of the selected column
        if col is not None:
            text = df[col].tolist()
            st.write(df[col].head())
else:

    # Create a text area for user to enter text and display the entered text
    text = st.text_area('Enter text:')
    st.text(text)
        
# Create a button to start
if st.button('Start'):
    with st.spinner('Processing...'):
        st.write(pd.DataFrame(clf(text)))
    st.success('Done')
