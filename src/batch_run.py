'''
This file contains the functions to run the NER pipeline on a batch of texts in a DataFrame.
'''

import pandas as pd
from typing import Dict, Any, List
import sys
import os

# Add the src directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .stelar_graph import chain


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
            
            # Extract results and add to DataFrame
            for output_field, column_name in output_columns.items():
                if output_field in result:
                    results_df.at[index, column_name] = result[output_field]
                    
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            # Set error values for this row
            for output_field, column_name in output_columns.items():
                results_df.at[index, column_name] = f"ERROR: {str(e)}"
    
    return results_df


def create_default_parameters() -> Dict[str, Any]:
    """
    Create a dictionary with default parameters for the NER pipeline.
    
    Returns:
        Dict[str, Any]: Default parameters dictionary
    """
    return {
        "translation_option": False,
        "ontology": [],
        "summary_option": True,
        "summary_model": "ollama:llama3.1:latest",
        "summary_base_url": None,
        "summary_model_instance": None,
        "main_entity_selection_option": True,
        "main_entity_selection_type": "single",
        "main_entity_model": "ollama:llama3.1:latest",
        "main_entity_base_url": None,
        "main_entity_model_instance": None,
        "ner_method": "LLM",
        "ner_model": "ollama:llama3.1:latest",
        "ner_base_url": None,
        "ner_custom_prompt": None,
        "ner_model_instance": None,
        "entity_linking_method": "LLM",
        "entity_linking_k": 3,
        "entity_linking_model": "ollama:llama3.1:latest",
        "entity_linking_base_url": None,
        "entity_linking_model_instance": None,
        "class_of_interest": "food",
        "evaluation_option": False
    }

