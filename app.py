# Import the required libraries
import streamlit as st 
import speech_recognition as sr
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the model path
model_path='Text2Poetry/model'

# Load the model path
def load_model(model_path):
    model=GPT2LMHeadModel.from_pretrained(model_path)
    return model

# Load the tokenizer
def load_tokenizer(tokenizer_path):
    tokenizer=GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

# Remove unnecessary digits and characters in output text
def remove_numbers(text):
 return ''.join(char for char in text if not char.isdigit())

def replace(text):
    return text.replace('\n','')

def cut(text):
   char_to_find = '"'
   start_index = text.find(char_to_find)
   if start_index != -1:
    substring = text[start_index:]
    return substring

# Function to generate content using GPT-2
def generate_text(model_path,sequence):
    model=load_model(model_path)
    tokenizer=load_tokenizer(model_path)
    IDs=tokenizer.encode(f'{sequence}',return_tensors='pt')
    outputs=model.generate(IDs,do_sample=True,max_length=50,pad_token_id=model.config.eos_token_id,top_k=50,top_p=0.97)
    res=tokenizer.decode(outputs[0],skip_special_tokens=True)
    res=replace(res)
    res=remove_numbers(res)
    res=cut(res)
    return res

# Initialize the streamlit app
st.set_page_config('Text2Poetry Generation')

st.title("Let's Enjoy Learning!!")

subheader_text = "Put your concept like definition, fact related to chemistry, biology and physics in the box below :"
style = f"<p style='font-size:18px;font-family:serif'> {subheader_text} </p>"
st.markdown(style, unsafe_allow_html=True)

subheader_text = "e.g. Reaction of hydrogen and oxygen forms water"
style = f"<p style='font-size:16px;font-family:serif'> {subheader_text} </p>"
st.markdown(style, unsafe_allow_html=True)

input=st.text_area('Word Limit : 25')

# Function to record audio 
def record_audio():
   rec=sr.Recognizer()
   with sr.Microphone() as mic:
        st.write('Listening...')
        audio=rec.listen(mic)
   try:
        text=rec.recognize_google(audio)
        return text
   except sr.UnknownValueError:
        st.error('Speak again')
   except sr.RequestError as re:
        st.error(re)
        
       
submit=st.button('Generate')
speak=st.button('Speak concept:microphone:')

# if speak button clicked
if speak:
    text=record_audio()
    st.text_input(label='Your text',value=text)
    res=generate_text(model_path,text)
    st.write(res)

# If submit button clicked
if submit:
    result=generate_text(model_path,input)
    remove_numbers(result)
    st.write(result)





