from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
import chromadb
import bm25s
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

def LLM_chooser(input_entity, list_items, k, chooser_model, base_url = None, model_instance = None) -> list[str]:
    """
    Chooses the k best matches for input_entity from list_items by prompting either the chooser_model or a model instance.

    Args: 
        input_entity (str): entity to get best matches for
        list_items (list[str]): list of entities to select best matches from
        k (int): number of best matches to select
        chooser_model (str): an llm supported by Langchain that chooses. If not pulled, need to pull it on Ollama
        model_instance (chat_model): instance of an llm to be used instead of instatiating a new one
    
    Returns: 
        entities_linked (list[str]): list of entities linked to input entity
    """
    if model_instance is None:
        model = init_chat_model(model=chooser_model, base_url = base_url, temperature=0, max_tokens=None)
    else:
        model = model_instance
    
    if k == 1:
        prompt_final = """
        You are given an input entity and a list of entities. Choose one entity from the list that best describes the product. 
        Please only return the list item and no additional text.
        \nInput entity: """ + input_entity + "\nList: " + str(list_items) + "\nAnswer:"
    else:
        prompt_final = """
        You are given an input entity and a list of entities. Choose strictly {k} entities from the given list that best describe the product. 
        Please only return a list of length {k} containing the {k} items chosen, and no additional text. If you return an item that does not belong to the list, you will be penalized.
        \nInput entity: {input_entity} "\nList: " + {list_items} + "\nAnswer: """
    input_variables = ["k"]
    prompt = PromptTemplate(
        template=prompt_final, input_variables=input_variables
    )
    # Chain
    chain = prompt | model
    entities_linked = chain.invoke({"k":k, "input_entity": input_entity, "list_items":list_items}).content[2:-2].split("', '")

    #If model returned an entity that does not belong to input_list, delete it from entities_linked
    print("entities linked before:", entities_linked)
    for item in entities_linked:
        if item not in list_items:
            entities_linked.remove(item)
    print("entities linked after filtering:", entities_linked)
    return entities_linked

def entity_linking_chromadb(input_entity, ontology, k, augm: bool = False, model_augm=None, base_url = None, model_instance = None) -> list[str]:
    """
    Links input entity to k closest ontology entities using ChromaDB. If augm=True, 5 instead of k entities are selected using ChromaDB and model_augm selects the k best matches among them.

    Args: 
        input_entity (str): entity to be linked
        ontology (list[str]): ontology items to link input entity to
        k (int): number of entities to be linked to input entity
        model_augm (str): an llm supported by Langchain that selects k best matches, in case augm=True. If not pulled, need to pull it on Ollama
        model_instance (chat_model): instance of an llm to be used instead of instatiating a new one
    
    Returns: 
        entities_linked (list[str]): list of entities linked to input entity
    """
    if augm:
        k_extr = 5
    else:
        k_extr = k
    client = chromadb.PersistentClient(path="../db/")
    try:
        collection = client.get_collection("AK_products")
    except:
        collection = client.create_collection("AK_products")       #create collection and insert ontology embeddings
        list_ids = [str(id) for id in range(len(ontology))]
        print("Creating collection and adding products...")
        collection.add(
            documents = ontology,
            ids = list_ids
        )
        print("Adding products completed")
    entities_linked = collection.query(    #query ontology
        query_texts=input_entity,
        n_results=k_extr
    )['documents'][0]
    if augm:
        print(model_augm)
        # entities_linked = LLM_chooser(input_entity=input_entity, list_items=entities_linked, k=k, chooser_model=model_augm, base_url = base_url, model_instance = model_instance)
        entities_linked = entity_linking_LLM(input_entity=input_entity, ontology=entities_linked, k=k, model_name="groq:llama-3.1-8b-instant", base_url = base_url, model_instance = model_instance)
    return entities_linked

def entity_linking_bm25s(input_entity, ontology, k, augm, model_augm=None, base_url = None, model_instance = None) -> list[str]:
    """
    Links input entity to k closest ontology entities using bm25s library. If augm=True, 5 instead of k entities are selected using bm25s and model_augm selects the k best matches among them.

    Args: 
        input_entity (str): entity to be linked
        ontology (list[str]): ontology items to link input entity to
        k (int): number of entities to be linked to input entity
        model_augm (str): an llm supported by Langchain that selects k best matches, in case augm=True. If not pulled, need to pull it on Ollama
        model_instance (chat_model): instance of an llm to be used instead of instatiating a new one
    
    Returns: 
        entities_linked (list[str]): list of entities linked to input entity
    """
    retriever = bm25s.BM25(corpus=ontology)
    retriever.index(bm25s.tokenize(ontology))
    results, scores = retriever.retrieve(bm25s.tokenize(input_entity), k=k)
    print("results bm25s:", results)
    entities_linked = list(results[0])
    if augm:
        print("Im here")
        # entities_linked = LLM_chooser(input_entity=input_entity, list_items=entities_linked, k=k, chooser_model=model_augm, base_url = base_url, model_instance = model_instance)
        entities_linked = entity_linking_LLM(input_entity=input_entity, ontology=entities_linked, k=k, model_name="groq:llama-3.1-8b-instant", base_url = base_url, model_instance = model_instance)
    return entities_linked

def entity_linking_LLM(input_entity, ontology, k, model_name, base_url = None, model_instance = None):
    """
    Links input entity to k closest ontology entities by prompting Ollama's model_name.

    Args: 
        input_entity (str): entity to be linked
        ontology (list[str]): ontology items to link input entity to
        k (int): number of entities to be linked to input entity
        model_name (str): ollama model that selects k best matches. If not pulled, need to pull it on Ollama
        model_instance (chat_model): instance of an llm to be used instead of instatiating a new one
    
    Returns: 
        entities_linked (list[str]): list of entities linked to input entity
    """
    if model_instance is None:
        print("Im here 2")
        model = init_chat_model(model=model_name, base_url = base_url, temperature=0, max_tokens=None)
    else:
        print("Im here 3")
        model = model_instance

    class LinkedEntities(BaseModel):
        """The linked entities from the ontology."""
        linked_entities: List[str]

    entity_linking_system_prompt = '''
    You are given a product and a list of food items. Choose exactly {k} items from the list that best describes the product. 
    Please only return the items and no additional text. If there is no similar item in the list, return empty list. 
    If you return an item that does not belong to the list, you will be penalized.
    '''
    # Example:
    # Product: Kinder Surprise Chocolate Egg
    # List: ["milk and confectionary", "chocolate", "chocolate bar", "chocolate egg"]
    # Answer: ["chocolate egg"]'''

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", entity_linking_system_prompt),
            ("human", "Product: {input_entity}\nList: {ontology}\nAnswer:"),
        ]
    )

    chain = prompt | model.with_structured_output(LinkedEntities)
    entities_linked = chain.invoke({"input_entity": input_entity, "ontology": ontology, "k": k})
    print("linked_entities:", entities_linked.linked_entities)

    # batch_size = 100
    # candidates = []
    # for batch_id in range(0, int(len(ontology)/batch_size)):
    #     upper_limit = (batch_id+1)*batch_size
    #     if upper_limit >= len(ontology):
    #         upper_limit = len(ontology) - 1
    #     basic_prompt = """
    #     You are given a product and a list of food items. Choose an item from the list that best describes the product. 
    #     Please only return the item and no additional text. If there is no similar item in the list, return "Nothing". If you return an item that does not belong to the list, you will be penalized.
    #     Product: """ + input_entity + ", list: " + str(ontology[batch_id*batch_size:upper_limit])
    #     answer = model.invoke(basic_prompt).content
    #     print("answer:", answer)
    #     if answer.lower() in ontology:
    #         candidates.append(answer)
    # entities_linked = LLM_chooser(input_entity, candidates, k, model_name, base_url = base_url, model_instance = model_instance)
    return entities_linked.linked_entities