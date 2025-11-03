from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import List
import pandas as pd 
import time
import re
import os
from dotenv import load_dotenv

load_dotenv()

def entity_extraction_llm(text, entity_type, model = "ollama:llama3.1:latest", base_url = None, custom_prompt = None, model_instance = None):
    """
    Extracts all entities of requested type from given text, by prompting given model.

    Args: 
        text (str): text to annotate entities from
        entity_type (str): entity type to be annotated (person, organization, ...)
        model (str): ollama model that performs NER. If should be supported by Langchain init_chat_model
        base_url (str): base url of the llm server
        custom_prompt (str): custom prompt to use for entity extraction. If not provided, the default prompt will be used.
        model_instance (chat_model): instance of an llm to be used instead of instatiating a new one
    
    Returns: 
        entities_list (list[str]): list of entities of type "entity_type" extracted from text
    """
    
    class Annotation(BaseModel):
        """Annotations extracted from the input text."""
        entities: List[str]

    if custom_prompt is None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                """You are a helpful assistant that identifies entities of type: {entity_type} in the a text.
                You have to identify the exact substrings of the text that correspond to the entities of this and only this type.
                If you identify multiple entities, separate them with a comma in a list.
                If you identify no entities, respond with an empty list.
                Respond with the following structure:
                entities: [entity1, entity2, ...] (a list not a str)
                
                YOUR TEXT:"""
            ),
            ("human", "{text}"),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",custom_prompt),
                ("human", "{text}"),
            ]
        )
    # Chain
    if model_instance is None:
        llm = init_chat_model(model=model, base_url = base_url, temperature=0, max_tokens=None)
    else:
        llm = model_instance
    try:
        structured_model = llm.with_structured_output(Annotation)
        chain = prompt | structured_model
        entities_list = []
        answer = chain.invoke({"text":text,"entity_type":entity_type})
        entities_list = answer.entities
        print(entities_list)
    # --- FALLBACK TO PLAIN TEXT EXTRACTION ---
    except Exception as e:
        print(f"⚠️ Structured output failed: {e}\nSwitching to fallback plain-text mode.")

        fallback_prompt = f"""
        You are an assistant that extracts entities of type "{entity_type}" from text.
        Extract the exact mentions as they appear in the text.
        Respond ONLY with a comma-separated list of entities (no JSON, no explanations).
        If none are found, return an empty string.

        Example output:
        entity1, entity2, entity3

        Text:
        {text}
        """

        # Simpler non-structured chain
        fallback_chain = ChatPromptTemplate.from_template(fallback_prompt) | llm
        raw_response = fallback_chain.invoke({"text": text})
        entities_list = re.split(",|, | ,", raw_response.content)
    return entities_list

def convert_entities_to_list(text, entities: list[dict]) -> list[str]:
        """
        Helper function for entity_extraction_ifroberta(text). Converts model output to list of food entities.

        Args: 
            text (str): text to retrieve the textual annotations from
            entities (list[dict]): list of entities extracted from text
        
        Returns: 
            entities_list (list[str]): list of entities of type "food" extracted from text
        """
        ents = []
        for ent in entities:
            e = {"start": ent["start"], "end": ent["end"], "label": ent["entity_group"]}
            if ents and -1 <= ent["start"] - ents[-1]["end"] <= 1 and ents[-1]["label"] == e["label"]:
                ents[-1]["end"] = e["end"]
                continue
            ents.append(e)

        return [text[e["start"]:e["end"]] for e in ents]

def entity_extraction_ifroberta(text):
    """
    Extracts all entities of type "food" from given text, using InstaFoodRoBERTa-ner (https://huggingface.co/Dizex/InstaFoodRoBERTa-NER).

    Args: 
        text (str): text to annotate entities from
    
    Returns: 
        entities_list (list[str]): list of entities of type "food" extracted from text
    """
    tokenizer = AutoTokenizer.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
    model = AutoModelForTokenClassification.from_pretrained("Dizex/InstaFoodRoBERTa-NER")

    pipe = pipeline("ner", model=model, tokenizer=tokenizer)

    ner_entity_results = pipe(text, aggregation_strategy="simple")
    ner_entity_results = convert_entities_to_list(text, ner_entity_results)
    return ner_entity_results