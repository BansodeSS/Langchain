from dotenv import load_dotenv
#load all the environment variable from .env
import streamlit as st
import os

from PIL import Image
import google.generativeai as genai
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Function to load Gemini Pro Vision

model = genai.GenerativeModel('gemini-pro-vision')
def get_gemini_response(input,image,prompt):
    response = model.generate_content([input,image[0],prompt])
    return response.text

def input_image_setup(upload_file):
    if upload_file is not None:
        bytes_data = upload_file.getvalue()
        image_parts = [
            {
            "mime_type": upload_file.type,
            "data":bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file Uploaded")



st.set_page_config(page_title="Multi-Language Invoice Extractor")
st.header("MultiLanguage Invoice Extractor")
input = st.text_input("Input Prompt:",key="input")
upload_file = st.file_uploader("Choose an image of the Invoice",type=["jpg","jpeg","png"])
image =""
if upload_file is not None:
    image= Image.open(upload_file)
    st.image(image,caption="Upload Image",use_column_width=True)
submit = st.button("Tell me about invoice")
input_prompt ='''
You are an expert in understanding invoices. We will upload a image as Invoice
and you will have to anser any questions based on the upload invoice image
'''
if submit:
    image_data = input_image_setup(upload_file)
    response = get_gemini_response(input_prompt,image_data,input)
    st.header("The response is")
    st.write(response)
