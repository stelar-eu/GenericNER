import time
import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

def translate_deep_translator(text):
    """
    Translate text in English using Google Translator via deep-translator API.
    text (str): text to translate
    """
    translation = GoogleTranslator(source='auto', target='en').translate(text)
    return translation

def translate_llm(text, model = "groq:llama-3.1-8b-instant", base_url = None, model_instance = None):
    """
    Translate text in English using an LLM.
    text (str): text to translate
    """
    if model_instance is None:
        model = init_chat_model(model=model, base_url = base_url, temperature=0, max_tokens=None)
    else:
        model = model_instance

    # model = ChatOllama(
    #     model="llama3.1:latest",
    #     base_url = 'http://localhost:11434',
    #     temperature=0,
    #     max_tokens=None,
    #     keep_alive=-1, 
    # )
    translation_prompt = "Translate the following text in English and print only the translation with no additional text. Text:" 
    translation = model.invoke(translation_prompt + text).content
    return translation

def translate_text(text, method = "deep-translator", model = "groq:llama-3.1-8b-instant", base_url = None, model_instance = None) -> str:
    """
    Translate text in English using given method.
    text (str): text to translate
    method (str): method to translate text. Supported methods: deep-translator, LLM
    """
    supported_methods = ["LLM", "deep-translator"]
    if method not in supported_methods:
        raise Exception("Translation method not supported. Supported methods: " + str(supported_methods))
    if method == "deep-translator":
        return translate_deep_translator(text)
    else:
        return translate_llm(text, model, base_url, model_instance)
    