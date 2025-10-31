from typing_extensions import TypedDict
from typing import List, Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from archive.functions import translate_text, summarize, main_entity_selection, entity_extraction, entity_linking
from archive.evaluation import evaluate_ner_list
import os
import pandas as pd
# Definition of the Graph State

# Graph State 
class State(TypedDict):
    '''
    context (str): The context of the document to be annotated.
    summary_option (bool): Whether the user wants to summarize the document.
    main_entity_selection_option (bool): Whether the user wants to select the main entity of the document.
    class_of_interest (str): The class of interest of the document for which ner will be performed.
    annotations (List[str]): The annotations of the document.
    linked_annotations (dict): The linked annotations of the document.
    evaluation_option (bool): Whether the user wants to evaluate the document.
    '''
    context: str
    translation_option: bool
    ontology: List[str]
    summary: str
    summary_option: bool 
    summary_model: str
    summary_base_url: str
    summary_model_instance: str
    main_entity_selection_option: bool
    main_entity_selection_type: str
    main_entity_model: str
    main_entity_base_url: str
    main_entity_model_instance: str
    main_entities: List[str]
    ner_method: str
    ner_model: str
    ner_base_url: str
    ner_custom_prompt: str
    ner_model_instance: str
    entity_linking_method: str
    entity_linking_k: int
    entity_linking_model: str
    entity_linking_base_url: str
    entity_linking_model_instance: str
    class_of_interest: str
    annotations: List[str]
    linked_annotations: dict
    evaluation_option: bool


# Nodes 
# The nodes are the main components of the graph that willl be executed during the invokation of the graph. 

def translator(state: State):
    """
    Translates the context of the document.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Either returns the original state or the state with a new key 'context' with the translated context.
    """
    if state['translation_option']:
        return {'context': translate_text(state['context'])}
    else:
        return state

def summarizer(state: State):
    """
    Summarizers the context of the document.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Either returns the original state or the state with a new key 'summary' with the summary of the context.
    """
    if state['summary_option']:
        print('summarizing')
        summary = summarize(state['context'], model = state['summary_model'], base_url = state['summary_base_url'], model_instance = state['summary_model_instance'])
        return {'summary': summary}
    else:
        return state

def perform_ner(state: State):
    """
    Performs NER and identifies all the entities of a given type in the text.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Adds a new key 'annotations' to the state with the entities of the class of interest.
    """

    response = entity_extraction(type = state["class_of_interest"], 
                                 text = state["context"],
                                 method = state["ner_method"],
                                 model = state["ner_model"],
                                 base_url = state["ner_base_url"],
                                 custom_prompt = state["ner_custom_prompt"],
                                 model_instance = state["ner_model_instance"])
    entities = response

    return {"annotations": entities}

def main_entity_selector(state: State):
    """
    Performs main entity selection and identifies which annotation(s) should be selected as the main entity.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Alters the value of the key 'annotations' to the main entity by reducing the initial list. 
    """
    if state['main_entity_selection_option']:
        main_entity = main_entity_selection(state['context'], 
                                            state['class_of_interest'], 
                                            state['annotations'], 
                                            selection_type = state['main_entity_selection_type'], 
                                            model = state['main_entity_model'], 
                                            base_url = state['main_entity_base_url'], 
                                            model_instance = state['main_entity_model_instance'])
        print(main_entity)
        return {'main_entities': main_entity}
    else:
        return state
    

def entity_linker(state: State):
    """
    Performs entity linking and links each entity to the best match(es) in a given ontology or list of classes.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Adds a new key 'linked_annotations' to the state with the linked annotations.
        The key is a dictionary where the keys are the entities and the values are lists containing the linked classes in descending order of confidence.
    """
    # check if main_entites have been set in the state  
    if 'main_entities' not in state:
        entities = state["annotations"]
    else:
        if (type(state["main_entities"]) == list) and (type(state["annotations"]) == list):
            entities = list(set(state["main_entities"] + state["annotations"]))
        else:
            entities = list(set(state["main_entities"].split() + state["annotations"]))
    
    linked_entities = {}
    counter = 0 
    for entity in entities:
        linked_entities[entity] = entity_linking(input_entity = entity, 
                                                ontology = state['ontology'], 
                                                k = state['entity_linking_k'], 
                                                method = state['entity_linking_method'], 
                                                model = state['entity_linking_model'], 
                                                model_augm = False,
                                                base_url = state['entity_linking_base_url'], 
                                                model_instance = state['entity_linking_model_instance'])
    
    return {"linked_annotations": linked_entities}

# def response_composer(state: State):
#     """Finally compose a response with all the annotations."""
    
#     response = ner_chain.invoke({
#         "entity_type": state["class_of_interest"],
#         "input": state["context"]
#     })

#     entities = response.content["entities"]

#     return {"annotations": entities}


# Graph Definition 

workflow = StateGraph(State)

# Addition of nodes belonging to the main agents in the workflow
workflow.add_node("translator", translator)
workflow.add_node("summarizer", summarizer)
workflow.add_node("ner", perform_ner)
workflow.add_node("main_entity_selector", main_entity_selector)
workflow.add_node("entity_linker", entity_linker)

# Add edges
workflow.add_edge(START, "translator")
workflow.add_edge("translator", "summarizer")
workflow.add_edge("summarizer", "ner")
workflow.add_edge("ner", "main_entity_selector")
workflow.add_edge("main_entity_selector", "entity_linker")
workflow.add_edge("entity_linker", END)

chain = workflow.compile()

# # Show workflow
# display(Image(chain.get_graph().draw_mermaid_png()))

# def test_run(json):
#     """
#     Performs a test run of the pipeline.
#     """
#     res = chain.invoke({"context": json['parameters']['context'], 
# 				"class_of_interest": json['parameters']['class_of_interest'],
# 				"summary_option": json['parameters']['summary_option'],
# 				"summary_model": json['parameters']['summary_model'],
# 				"summary_base_url": json['parameters']['summary_base_url'],
# 				"summary_model_instance": json['parameters']['summary_model_instance'],
# 				"main_entity_selection_option": json['parameters']['main_entity_selection_option'],
# 				"main_entity_selection_type": json['parameters']['main_entity_selection_type'],
# 				"main_entity_model": json['parameters']['main_entity_model'],
# 				"main_entity_base_url": json['parameters']['main_entity_base_url'],
# 				"main_entity_model_instance": json['parameters']['main_entity_model_instance'],
# 				"ner_method": json['parameters']['ner_method'],
# 				"ner_model": json['parameters']['ner_model'],
# 				"ner_base_url": json['parameters']['ner_base_url'],
# 				"ner_custom_prompt": json['parameters']['ner_custom_prompt'],
# 				"ner_model_instance": json['parameters']['ner_model_instance'],
# 				"entity_linking_method": json['parameters']['entity_linking_method'],
# 				"entity_linking_model": json['parameters']['entity_linking_model'],
# 				"entity_linking_base_url": json['parameters']['entity_linking_base_url'],
# 				"entity_linking_model_instance": json['parameters']['entity_linking_model_instance'],
# 				"ontology": json['parameters']['ontology'],
# 				"entity_linking_k": json['parameters']['entity_linking_k'],
# 				"entity_linking_augm": json['parameters']['entity_linking_augm'],
# 				"entity_linking_model_augm": json['parameters']['entity_linking_model_augm']
# 			})
#     print(res)
#     return res
    

def front_end_run(
        context: str,
        class_of_interest: str,
        summary_option: bool,
        summary_model: str,
        main_entity_selection_option: bool,
        main_entity_selection_type: str,
        main_entity_model: str,
        ner_method: str,
        ner_model: str,
        entity_linking_method: str,
        entity_linking_model: str,
        entity_linking_k: int,
        entity_linking_augm: bool,
        entity_linking_model_augm: str,
        ontology: List[str],
        translation_option: bool = True,
        translation_method: str = "deep-translator",
    ):
    """
    Performs a test run of the pipeline.
    """
    res = chain.invoke({"context": context, 
                        "translation_option": translation_option,
                        "translation_method": translation_method,
                  "class_of_interest": class_of_interest,
                  "summary_option": summary_option,
                  "summary_model": summary_model,
                  "summary_base_url": None,
                  "summary_model_instance": None,
                  "main_entity_selection_option": main_entity_selection_option,
                  "main_entity_selection_type": main_entity_selection_type,
                  "main_entity_model": main_entity_model,
                  "main_entity_base_url": None,
                  "main_entity_model_instance": None,
                  "ner_method": ner_method,
                  "ner_model": ner_model,
                  "ner_base_url": None,
                  "ner_custom_prompt": None,
                  "ner_model_instance": None,
                  "entity_linking_method": entity_linking_method,
                  "entity_linking_model": entity_linking_model,
                  "entity_linking_base_url": None,
                  "entity_linking_model_instance": None,
                  "ontology": ontology,
                  "entity_linking_k": entity_linking_k,
                  "entity_linking_augm": entity_linking_augm,
                  "entity_linking_model_augm": entity_linking_model_augm
                  })
    return res

def test_run():
    """
    Performs a test run of the pipeline.
    """
    res = chain.invoke({"context": "The quick brown fox jumps over the lazy dog", 
                  "class_of_interest": "animal",
                  "summary_option": True,
                  "summary_model": "ollama:llama3.1:latest",
                  "summary_base_url": None,
                  "summary_model_instance": None,
                  "main_entity_selection_option": False,
                  "main_entity_selection_type": "single",
                  "main_entity_model": "ollama:llama3.1:latest",
                  "main_entity_base_url": None,
                  "main_entity_model_instance": None,
                  "ner_method": "llm",
                  "ner_model": "ollama:llama3.1:latest",
                  "ner_base_url": None,
                  "ner_custom_prompt": None,
                  "ner_model_instance": None,
                  "entity_linking_method": "llm",
                  "entity_linking_model": "ollama:llama3.1:latest",
                  "entity_linking_base_url": None,
                  "entity_linking_model_instance": None,
                  "ontology": ["animal", "person", "food", "plant", "object"],
                  "entity_linking_k": 3,
                  "entity_linking_augm": False,
                  "entity_linking_model_augm": None
                  })
    print(res)
    return res

def test_run_groq():
    

    """
    Performs a test run of the pipeline.
    """
    res = chain.invoke({"context": "The quick brown fox jumps over the lazy dog", 
                  "class_of_interest": "animal",
                  "summary_option": True,
                  "summary_model": "groq:llama-3.1-8b-instant",
                  "summary_base_url": None,
                  "summary_model_instance": None,
                  "main_entity_selection_option": False,
                  "main_entity_selection_type": "single",
                  "main_entity_model": "groq:llama-3.1-8b-instant",
                  "main_entity_base_url": None,
                  "main_entity_model_instance": None,
                  "ner_method": "llm",
                  "ner_model": "groq:llama-3.1-8b-instant",
                  "ner_base_url": None,
                  "ner_custom_prompt": None,
                  "ner_model_instance": None,
                  "entity_linking_method": "llm",
                  "entity_linking_model": "groq:llama-3.1-8b-instant",
                  "entity_linking_base_url": None,
                  "entity_linking_model_instance": None,
                  "ontology": ["animal", "person", "food", "plant", "object"],
                  "entity_linking_k": 3,
                  "entity_linking_augm": False,
                  "entity_linking_model_augm": None
                  })
    print(res)
    return res

# def test_run():
#     """
#     Performs a test run of the pipeline.
#     """
#     res = chain.invoke({"context": "The quick brown fox jumps over the lazy dog", 
#                   "class_of_interest": "animal",
#                   "summary_option": False,
#                   "summary_model": "groq:llama-3.1-8b-instant",
#                   "summary_base_url": None,
#                   "summary_model_instance": None,
#                   "main_entity_selection_option": False,
#                   "main_entity_selection_type": "single",
#                   "main_entity_model": "groq:llama-3.1-8b-instant",
#                   "main_entity_base_url": None,
#                   "main_entity_model_instance": None,
#                   "ner_method": "llm",
#                   "ner_model": "groq:llama-3.1-8b-instant",
#                   "ner_base_url": None,
#                   "ner_custom_prompt": None,
#                   "ner_model_instance": None,
#                   "entity_linking_method": "llm",
#                   "entity_linking_model": "ollama:llama3.1:latest",
#                   "entity_linking_base_url": None,
#                   "entity_linking_model_instance": None,
#                   "ontology": ["animal", "person", "food", "plant", "object"],
#                   "entity_linking_k": 1,
#                   "entity_linking_augm": False,
#                   "entity_linking_model_augm": None
#                   })
#     print(res)
#     return res


def batch_run_ner(df: pd.DataFrame, 
                  text_column: str,
                  parameters: Dict[str, Any],
                  output_columns: Dict[str, str] = None) -> pd.DataFrame:
    """
    Run NER pipeline on a batch of texts in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing texts to process
        text_column (str): Name of the column containing the text to process
        parameters (Dict[str, Any]): Dictionary containing all pipeline parameters
        output_columns (Dict[str, str]): Optional mapping of output field names to column names
                                       Default: {'annotations': 'annotations', 'linked_annotations': 'linked_annotations', 
                                               'summary': 'summary', 'main_entities': 'main_entities'}
    
    Returns:
        pd.DataFrame: Original DataFrame with additional columns for NER results
    """
    if output_columns is None:
        output_columns = {
            'annotations': 'annotations',
            'linked_annotations': 'linked_annotations', 
            'summary': 'summary',
            'main_entities': 'main_entities'
        }
    
    # Validate input
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Initialize result columns
    results_df = df.copy()
    for output_field, column_name in output_columns.items():
        results_df[column_name] = None
    
    total_errors = 0 
    total_annotated_rows = 0 
    # Process each row
    for index, row in df.iterrows():
        try:
            text = row[text_column]
            
            # Prepare input for the pipeline
            pipeline_input = {
                "context": text,
                **parameters
            }
            
            # Run the pipeline
            result = chain.invoke(pipeline_input)
            total_annotated_rows += 1
            
            # Extract results and add to DataFrame
            for output_field, column_name in output_columns.items():
                if output_field in result:
                    results_df.at[index, column_name] = result[output_field]
                    
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            total_errors += 1
            # Set error values for this row
            for output_field, column_name in output_columns.items():
                results_df.at[index, column_name] = f"ERROR: {str(e)}"
    
    if 'annotations' in results_df.columns:
        annotations_per_row = results_df['annotations'].apply(len).mean()
    else:
        annotations_per_row = 0

    metrics_dict = {
        "total_errors": total_errors,
        "total_annotated_rows": total_annotated_rows,
        "annotations_per_row": annotations_per_row
    }
    return results_df, metrics_dict  



if __name__ == "__main__":

    context = "The quick brown fox jumps over the lazy dog"
    class_of_interest = "animal"
    summary_option = True
    summary_model = "groq:llama-3.1-8b-instant"
    main_entity_selection_option = False
    main_entity_selection_type = "single"
    translation_option = True
    translation_method = "deep-translator"
    main_entity_model = "groq:llama-3.1-8b-instant"
    ner_method = "llm"
    ner_model = "groq:llama-3.1-8b-instant"
    entity_linking_method = "llm"
    entity_linking_model = "groq:llama-3.1-8b-instant"
    entity_linking_k = 3
    entity_linking_augm = False
    entity_linking_model_augm = None
    ontology = ["animal", "person", "food", "plant", "object"]

    parameters = {              
        "translation_option": translation_option,
        "translation_method": translation_method,
        "class_of_interest": class_of_interest,
        "summary_option": summary_option,
        "summary_model": summary_model,
        "summary_base_url": None,
        "summary_model_instance": None,
        "main_entity_selection_option": main_entity_selection_option,
        "main_entity_selection_type": main_entity_selection_type,
        "main_entity_model": main_entity_model,
        "main_entity_base_url": None,
        "main_entity_model_instance": None,
        "ner_method": ner_method,
        "ner_model": ner_model,
        "ner_base_url": None,
        "ner_custom_prompt": None,
        "ner_model_instance": None,
        "entity_linking_method": entity_linking_method,
        "entity_linking_model": entity_linking_model,
        "entity_linking_base_url": None,
        "entity_linking_model_instance": None,
        "ontology": ontology,
        "entity_linking_k": entity_linking_k,
        "entity_linking_augm": entity_linking_augm,
        "entity_linking_model_augm": entity_linking_model_augm   
    }

    context_2 = "The children wanted to eat the chocolates before the parents arrived."

    # create a df with a column 'context'
    df = pd.DataFrame({'context': [context, context_2]})
    df['gold_standard'] = [['fox'], ['parents']]

    res_df, res_metrics = batch_run_ner(df, text_column = "context", parameters = parameters, output_columns = {'annotations': 'annotations', 'linked_annotations': 'linked_annotations'})
    evaluation_metrics = evaluate_ner_list(res_df, gold_standard = 'gold_standard', predictions = 'annotations')
    print(res_df)
    print(res_metrics)
    print(evaluation_metrics)

# def single_run(context, summary_option, main_entity_selection_option, class_of_interest):
#     """Performs the pipeline on a single string of text.
    
#     Args:
#         context (str): The context of the document to be annotated.
#         summary_option (bool): Whether the user wants to summarize the document.
#         main_entity_selection_option (bool): Whether the user wants to select the main entity of the document.
#         class_of_interest (str): The class of interest of the document for which ner will be performed.

#     Returns:
#         summary (str): The summary of the document.
#         main_entity (str): The main entity of the document.
#         linked_annotations (dict): The linked annotations of the document.
#     """
#     pass

# def batch_run(dataset, class_of_interest, summary_option, main_entity_selection_option, 
#               ner_method, summary_method, main_entity_method, entity_linking_method, 
#               ner_model, summary_model, main_entity_model, entity_linking_model, 
#               entity_linking_k, entity_linking_augm, entity_linking_model_augm):
#     """
#     Performs the pipeline on a batch of documents.

#     Args:
#         dataset (dataframe): A pandas dataframe with a column 'context' containing the documents to be processed.
#     Returns:
#         result_df (dataframe): A pandas dataframe with a copy of the intiial dataframe with the following columns:
#         - summary: The summary of the document.
#         - annotations: The annotations of the document.
#         - main_entity: The main entity of the document.
#         - linked_annotations: The linked annotations of the document.
#     """
#     result_df = dataset.copy()
#     result_df['summary'] = None
#     result_df['main_entity'] = None
#     result_df['linked_annotations'] = None

#     for index, row in result_df.iterrows():
#         tmp_summary, tmp_annotations, tmp_main_entity, tmp_linked_annotations = single_run(row['context'], row['summary_option'], row['main_entity_selection_option'], row['class_of_interest'])
#         result_df.at[index, 'summary'] = tmp_summary
#         result_df.at[index, 'annotations'] = tmp_annotations
#         result_df.at[index, 'main_entity'] = tmp_main_entity
#         result_df.at[index, 'linked_annotations'] = tmp_linked_annotations
#     return result_df

# def evaluate_run(results_df, ground_truth_df, evaluation_method = "simple", evaluator_model = "llama3.1:latest"):
#     """
#     Evaluates the results of the pipeline.

#     Args:
#         results_df (dataframe): A pandas dataframe with the results of the pipeline.
#         ground_truth_df (dataframe): A pandas dataframe with the ground truth of the pipeline.
#     Returns:
#         metrics (dict): A dictionary with the following keys:
#         - precision_strict
#         - accuracy_strict
#         - f1_score_strict
#         - precision_flexible
#         - accuracy_flexible
#         - f1_score_flexible
#         - total_succesfully_evaluated_rows
#     """
#     return "Not implemented"

# if __name__ == "__main__":
#     # test_run()

    
