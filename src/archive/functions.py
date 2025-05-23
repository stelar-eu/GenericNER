from pydantic import BaseModel, Field
from typing import List
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from .entity_linking_lib import entity_linking_chromadb, entity_linking_bm25s, entity_linking_LLM
from .entity_extraction_lib import entity_extraction_llm, entity_extraction_ifroberta
from .translation_lib import translate_deep_translator, translate_llm
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

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

def summarize(text, model = "ollama:llama3.1:latest", base_url = None, model_instance = None, custom_prompt = None) -> str:
    """
    Summarizes given text by prompting a given llm.

    Args: 
        text (str): text to summarize
        model (str): llm model that performs summarization (should be supported by Langchain init_chat_model)
        base_url (str): base url of the llm server
        model_instance (chat_model): instance of an llm to be used instead of instatiating a new one
        custom_prompt (str): custom prompt to use for summarization. If not provided, the default prompt will be used.
    Returns: 
        summary (str): summary of the text
    """
    if model_instance is None:
        model = init_chat_model(model=model, base_url = base_url, temperature=0, max_tokens=None)
    else:
        model = model_instance

    if custom_prompt is None:
        summary_system_prompt = "You are a helpful assistant that summarizes text. You are given a text and you need to summarize it in a very brief manner, making sure not to miss any important information. Print only the summary and no additional text."
    else:
        summary_system_prompt = custom_prompt

    summary_prompt = ChatPromptTemplate(
        messages=[
            ("system", summary_system_prompt),
            ("user", "Text to summarize: {text}")
        ]
    )
    summary_chain = summary_prompt | model
    summary = summary_chain.invoke({"text": text}).content
    return summary


def main_entity_selection(text, type, input_list, selection_type= "single", model = "ollama:llama3.1:latest", base_url = None, model_instance = None) -> str:
    """
    Selects main entity from the list given. The list contains entities of type "type" in given text. Entity selection is done by prompting a given llm.

    Args: 
        text (str): text to select main entity from
        type(str): entity type (e.g. food, person, ...)
        input_list (list[str]): list of entities of type "type" extracted from text.
        selection_type (str): type of main entity selection. Current supported options: single, multiple
        model (str): llm model that performs main entity selection (should be supported by Langchain init_chat_model)
        base_url (str): base url of the llm server
        model_instance (chat_model): instance of an llm to be used instead of instatiating a new one
    
    Returns: 
        main_entity (str): main entity selected from the list
    """
    if model_instance is None:
        model = init_chat_model(model=model, base_url = base_url, temperature=0, max_tokens=None)
    else:
        model = model_instance

    # init a pydantic model to request structured output from the model
    class EntitiesList(BaseModel):
        main_entities: List[str] = Field(description="main entities selected from list", required=True)

    
    # select between main entity selection options
    if selection_type == "single":
        input_variables = ["text", "input_list", "type"]
        prompt = """You are given a list with entities of type {type} identified in a text you will be given.
        Select one entity from that list which you think the text mainly refers to. Print only the main entity selected without rephrasing it and print no additional text. 
        \nText: {text} \nList: {input_list} \nAnswer:"""
        prompt = PromptTemplate(
            template=prompt, input_variables=input_variables
        )
        structured_model = model.with_structured_output(EntitiesList)
        chain = prompt | structured_model
        res = chain.invoke({"text":text,"input_list": input_list, "type":type}).main_entities

    elif selection_type == "multiple": 
        input_variables = ["text", "input_list", "type"]
        prompt = """You are given a list with entities of type {type} identified in a text you will be given.
        Select all entities from that list which you think that constitute main entities in the text. 
        For example in the case of a text with the text cookies with strawberry flavour and chocolate flavour, the main entities are cookies. 
        Return only the main entities selected without rephrasing them in JSON format inside a list, where the key is "main_entities" and the value is the list of main entities.
        \nText: {text} \nList: {input_list} \nAnswer:"""
        prompt = PromptTemplate(
            template=prompt, input_variables=input_variables
        )
        structured_model = model.with_structured_output(EntitiesList)
        chain = prompt | structured_model
        res = chain.invoke({"text":text,"input_list": input_list, "type":type}).main_entities
    else: 
        raise Exception("Invalid selection type. Supported types: single, multiple")

    return res

def entity_extraction(type, text, method = "llm", model = "ollama:llama3.1:latest", base_url = None, custom_prompt = None, model_instance = None) -> list[str]:
    """
    Extracts all entities of requested type from given text, using llm and langchain's structured output or InstaFoodRoBERTa-ner (https://huggingface.co/Dizex/InstaFoodRoBERTa-NER).
    For the latter, the type to be extracted must be "food".

    Args: 
        type (str): entity type to be annotated (person, organization, ...)
        text (str): text to annotate entities from
        method (str): method to use for entity extraction. Current methods supported: LLM, InstaFoodRoBERTa
        model (str): llm model that performs entity extraction (should be supported by Langchain init_chat_model)
        base_url (str): base url of the llm server
        custom_prompt (str): custom prompt to use for entity extraction. If not provided, the default prompt will be used.
        model_instance (chat_model): instance of an llm to be used instead of instatiating a new one
    
    Returns: 
        entities_list (list[str]): list of entities of type "type" extracted from text
    """
    supported_methods = ["llm","instafoodroberta"]
    entities_list = []
    if method not in supported_methods:
        raise Exception("Entity extraction not supported. Supported methods: " + str(supported_methods))
    
    if method == "llm":
        entities_list = entity_extraction_llm(text = text, entity_type = type, model = model, base_url = base_url, custom_prompt = custom_prompt, model_instance = model_instance)
    elif method == "instafoodroberta":
        if type == "food":
            entities_list = entity_extraction_ifroberta(text = text)
        else:
            raise Exception("InstaFoodRoBERTa extracts only food entities. Please rerun with type=food." )
    return entities_list

def entity_linking(input_entity, ontology, k, method, model= "ollama:llama3.1:latest", model_augm= "ollama:llama3.1:latest", base_url = None, model_instance = None) -> list[str]:
    """
    Links input entity to k closest ontology entities.

    Args: 
        input_entity (str): entity to be linked
        ontology (list[str]): ontology items to link input entity to
        k (int): number of entities to be linked to input entity
        method (str): method to use for entity linking. Current methods supported: ChromaDB, bm25s, augmented ChromaDB, augmented bm25s, llm
        model (str): llm model that performs entity linking, in case LLM method is selected. The model should be supported by Langchain init_chat_model
        model_augm (str): llm model that augments chromadb or bm25s, in case chromadb_aug or bm25s_aug method is selected (see entity_linking_bm25s/entity_linking_chromadb functions). The model should be supported by Langchain init_chat_model
        model_instance (chat_model): instance of an llm to be used instead of instatiating a new one
    
    Returns: 
        entities_linked (list[str]): list of entities linked to input entity
    """
    supported_methods = ["chromadb", "bm25s", "chromadb_aug", "bm25s_aug", "llm", "no_linking"]
    entities_linked = []
    if method not in supported_methods:
        raise Exception("Entity extraction not supported. Supported methods: " + str(supported_methods))
    if method == "chromadb":
        entities_linked = entity_linking_chromadb(input_entity=input_entity, ontology=ontology, k=k, augm = False)
    elif method == "bm25s":
        entities_linked = entity_linking_bm25s(input_entity=input_entity, ontology=ontology, k=k, augm = False)
    elif method == "chromadb_aug":
        entities_linked = entity_linking_chromadb(input_entity=input_entity, ontology=ontology, k=k, augm = True, model_augm=model_augm, base_url = base_url, model_instance = model_instance)
    elif method == "bm25s_aug":
        entities_linked = entity_linking_bm25s(input_entity=input_entity, ontology=ontology, k=k, augm = True, model_augm=model_augm, base_url = base_url, model_instance = model_instance)
    elif method == "llm":
        entities_linked = entity_linking_LLM(input_entity=input_entity, ontology=ontology, k=k, model_name=model, base_url = base_url, model_instance = model_instance)
    elif method == "no_linking":
        entities_linked = [input_entity]
    return entities_linked

if __name__ == "__main__":
    # returned = main_entity_selection(
    #     text = "I went to the shop to buy tomatoes next to the cucumber but they were super expensive so I did not buy them finally", 
    #     type = "food", 
    #     input_list = ["tomatoes", "cucumber"], 
    #     selection_type = "single", 
    #     model = "ollama:llama3.1:latest"
    # )
    # print(returned, type(returned))

    returned = entity_extraction(type = "food", text = "George and Jane went to the museum and saw a painting of grapes", method = "llm", model = "groq:llama-3.1-8b-instant")
    print(returned, type(returned))

    # returned = entity_linking(input_entity = "tomato shaped breadsticks", ontology = ['bread alternatives', 'fruits', 'vegetables'], k = 2, method = "llm", model = "ollama:llama3.1:latest")
    # print(returned, type(returned))
#     ONTOLOGY_PATH = "/mnt/data/vpitsilou/products_clear.csv"
#     ontology = pd.read_csv(ONTOLOGY_PATH)["product_name"]

#     # food entity extraction using InstaFoodRoBERTa
#     returned = entity_extraction(type = "food", text = "George and Jane went to the museum and saw a painting of grapes", method = "instafoodroberta", model = "llama3.1:latest")
#     print(returned, type(returned))

#     #generic entity extraction using llama3.1
#     returned = entity_extraction(type = "person", text = "George and Jane went to the museum and saw a painting of grapes", method = "llm", model = "llama3.1:latest")
#     print(returned, type(returned))

#     df = pd.read_csv("incidents.csv")
#     df = df.drop_duplicates("description")
#     texts = list(df['description'])

#     #food entity extraction using llama3.1
#     returned = entity_extraction(type = "food", text = texts[0], method = "llm", model = "llama3.1:latest")
#     print(returned, type(returned))
    
#     #main entity selection using llama3.1
#     returned = main_entity_selection(text=texts[0], type = "food", input_list=returned, model = "llama3.1:latest")
#     print(returned, type(returned))

#     #summarization
#     returned = summarize(text="I went to the shop to buy tomatoes but they were super expensive so I did not buy them finally", model = "llama3.1:latest")
#     print(returned, type(returned))

#     #entity linking (chromadb)
#     returned = entity_linking(input_entity = "banana bread", ontology = list(ontology), k = 3, method = "chromadb")
#     print(returned, type(returned))

#     #entity linking (bm25s)
#     returned = entity_linking(input_entity = "banana bread", ontology = list(ontology), k = 3, method = "bm25s")
#     print(returned, type(returned))

#     #entity linking (chromadb + llama3.1)
#     returned = entity_linking(input_entity = "banana bread", ontology = list(ontology), k = 3, method = "chromadb_aug", model_augm="llama3.1:latest")
#     print(returned, type(returned))

#     #entity linking (bm25s + llama3.1)
#     returned = entity_linking(input_entity = "banana bread", ontology = list(ontology), k = 3, method = "bm25s_aug", model_augm="llama3.1:latest")
#     print(returned, type(returned))

#     #entity linking (llama3.1)
#     returned = entity_linking(input_entity = "banana bread", ontology = list(ontology), k = 3, method = "llm", model="llama3.1:latest")
#     print(returned, type(returned))