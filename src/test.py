import sys
import os

# Add the parent directory to the path so we can import the src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stelar_graph import batch_run_ner
from src.archive.evaluation import evaluate_ner_list
import pandas as pd

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
    output_columns = {'annotations': 'annotations', 'linked_annotations': 'linked_annotations'}

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

    res_df, res_metrics = batch_run_ner(df, text_column = "context", parameters = parameters, output_columns = output_columns)
    evaluation_metrics = evaluate_ner_list(res_df, gold_standard = 'gold_standard', predictions = 'annotations')
    print(res_df)
    print(res_metrics)
    print(evaluation_metrics)