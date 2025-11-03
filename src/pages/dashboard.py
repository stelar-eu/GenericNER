import streamlit as st
import sys
import os
import pandas as pd
from pathlib import Path
import asyncio
from streamlit_extras.annotated_text import annotated_text
from stelar_graph import test_run, front_end_run

def show():
    # Initialize saved_annotations_df as a session state variable
    if 'saved_annotations_df' not in st.session_state:
        if os.path.exists("saved_annotations_df.csv"):
            st.session_state.saved_annotations_df = pd.read_csv("saved_annotations_df.csv")
        else:
            st.session_state.saved_annotations_df = pd.DataFrame(columns=["text", "class_of_interest", "summary", "annotations", "main_entities", "linked_annotations"])

    if 'current_response' not in st.session_state:
        st.session_state.current_response = {}
        st.session_state.current_response['context'] = ""
        st.session_state.current_response['summary'] = ""
        st.session_state.current_response['annotations'] = []
        st.session_state.current_response['main_entities'] = []
        st.session_state.current_response['linked_annotations'] = {}

    def format_annotations_for_display(text, entities, entity_tag):
        # Sort entities by their position in the text to maintain order
        entities = sorted(entities, key=lambda x: text.find(x))
        
        annotations = []
        last_end = 0
        
        for entity in entities:
            start = text.find(entity, last_end)
            if start == -1:  # Entity not found
                continue
                
            # Add the text before the entity
            if start > last_end:
                annotations.append(text[last_end:start])
                
            # Add the entity with its annotation
            annotations.append((entity, entity_tag, "#ff0000"))
            
            last_end = start + len(entity)
        
        # Add any remaining text after the last entity
        if last_end < len(text):
            annotations.append(text[last_end:])
            
        return annotations

    async def execute_pipeline(context: str, analysis_types: list[str]) -> str:
        return "test"

    async def dummy_workflow(context: str, analysis_types: list[str]) -> dict:
        return {"context": context, "summary": "test summary", "annotations": ['test1', 'test2'], "linked_annotations": {'test1': ['test1.1', 'test1.2'], 'test2': ['test2.1', 'test2.2']}}

    available_models = ["ollama:llama3.1:latest", "ollama:mistral:v0.3", "groq:llama-3.1-8b-instant"]
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("Pipeline Configuration")
        st.write("Here you can configure the pipeline to your needs")
        st.divider()
        st.header("Translation Options")
        st.write("Do you want to translate the text?")
        translation_option = st.checkbox("Translate", key = "translation_option")
        if translation_option:
            st.write("Please select the model you want to use for translation:")
            translation_method = st.selectbox("Translation Method", ["LLM", "deep-translator"], key = "translation_method")
        st.divider()
        st.header("Summarization Options")
        st.write("Do you want to summarize the text?")
        summarize_option = st.checkbox("Summarize", key = "summarize_option")
        if summarize_option:
            st.write("Please select the model you want to use for summarization:")
            summarization_model = st.selectbox("Summarization Model", available_models, key = "summarization_model")
        st.divider()
        st.header("Main Entity Selection")
        st.write("Do you want to select the main entity from a list of entities?")
        main_entity_selection_option = st.selectbox("Main Entity Selection", ["No", "Single", "Multiple"], key = "main_entity_selection_option")
        if main_entity_selection_option in ["Single", "Multiple"]:
            st.write("Please select the model you want to use for main entity selection:")
            main_entity_selection_model = st.selectbox("Main Entity Selection Model", available_models, key = "main_entity_selection_model")
        st.divider()
        st.header("Named Entity Recognition Options")
        st.write("How do you want to extract named entities from the text?")
        ner_option = st.selectbox("NER Option", ["LLMs", "iFoodRoBERTa"], key = "ner_option")
        if ner_option == "LLMs":
            st.write("Please select the model you want to use for named entity recognition:")
            ner_model = st.selectbox("Model", available_models, key = "ner_model")
        st.divider()
        st.header("Entity Linking Options")
        st.write("Do you want to link the entities to an ontology?")

        st.write("Please upload a csv file with the ontology you want to use for entity linking:")
        ontology_file = st.file_uploader("Upload Ontology", type = "csv")
        if ontology_file:
            # read the ontology which is a single column csv file into a list
            ontology_list = pd.read_csv(ontology_file)['ontology'].tolist()
            st.write(ontology_list)
        st.write("Please select the method you want to use for entity linking:")
        el_method = st.selectbox("Method", ["LLM", "ChromaDB", "ChromaDB_aug", "BM25s", "BM25s_aug", "no_linking"], key = "el_method")
        if el_method != "no_linking":
            st.write("Please select k:")
            el_k = st.number_input("k", min_value = 1, max_value = 10, value = 3, key = "el_k")
        if el_method == "LLM":
            st.write("Please select the model you want to use for entity linking:")
            el_model = st.selectbox("Model", available_models, key = "el_model")
        elif el_method == "ChromaDB_aug" or el_method == "BM25s_aug":
            st.write("Please select the model you want to use to augment the entity linking:")
            el_model_aug = st.selectbox("Model", available_models, key = "el_model_aug")

    # Title and description
    st.title("STELAR NER - EL Dashboard")
    st.markdown("""
    This dasboard allows you to configure and run the STELAR NER and EL pipeline on an instance level and visualize the results. 
    Configure the pipeline in the sidebar and click the "Analyze" button to run the pipeline.
    Click `Save` to store the results in the bottom comparison panel. 
    """)

    input_col, output_col = st.columns(2)
    with input_col:
        # Main content area
        st.header("Input Text")
        text_input = st.text_area("Enter your text here:", height=150)
        st.divider()
        st.header("Class of Interest")
        class_of_interest = st.text_input("Enter the class of interest here:")
        st.divider()
        #  Button to trigger the analysis
        if st.button("Analyze"):
            print("summarize option:", summarize_option, ", mes_option:", (main_entity_selection_option != "No") )
            response = front_end_run(
                context = text_input,
                translation_option = translation_option,
                translation_method = translation_method if translation_option else "",
                class_of_interest = class_of_interest,
                summary_option = summarize_option,
                summary_model = summarization_model if summarize_option else "",
                main_entity_selection_option = (main_entity_selection_option != "No"),
                main_entity_selection_type = main_entity_selection_option.lower() if main_entity_selection_option != "No" else "",
                main_entity_model = main_entity_selection_model if main_entity_selection_option != "No" else "",
                ner_method = "llm" if ner_option == "LLMs" else "instafoodroberta",
                ner_model = ner_model if ner_option == "LLMs" else "",
                entity_linking_method = el_method.lower(),
                entity_linking_model = el_model if el_method == "LLM" else "",
                entity_linking_k = el_k if el_method != "no_linking" else "",  # You might want to add this as a UI option
                entity_linking_augm = False if el_method != "ChromaDB_aug" and el_method != "BM25s_aug" else True,  # You might want to add this as a UI option
                entity_linking_model_augm = el_model_aug if el_method == "ChromaDB_aug" or el_method == "BM25s_aug" else "",  # You might want to add this as a UI option
                ontology = ontology_list
            )
            if 'context' in response.keys():
                st.session_state.current_response['context'] = response['context']
            # response = test_run()
            if 'summary' in response.keys():
                st.session_state.current_response['summary'] = response['summary']
            if 'annotations' in response.keys():
                st.session_state.current_response['annotations'] = response['annotations']
            if "main_entities" in response.keys():
                st.session_state.current_response['main_entities'] = response['main_entities']
            if 'linked_annotations' in response.keys():
                st.session_state.current_response['linked_annotations'] = response['linked_annotations']

    with output_col:
        st.header("Output")
        st.write("There you can visualize the output of the pipeline!")
        # In your analysis section:
        st.subheader("Annotated Text")
        if st.session_state.current_response['context'] != "":
            formatted_annotations = format_annotations_for_display(st.session_state.current_response['context'], st.session_state.current_response['annotations'], class_of_interest)
            annotated_text(*formatted_annotations)

        st.subheader("Pipeline Results")
        st.write("Summary")
        st.write(st.session_state.current_response['summary'])
        st.write("Annotations")
        st.write(list(set(st.session_state.current_response['annotations'])))
        st.write("Main Entities")
        st.write(st.session_state.current_response['main_entities'])
        st.write("Linked Entities")
        st.write(st.session_state.current_response['linked_annotations'])
        save_button = st.button("Save")
        if save_button:
            new_row_index = len(st.session_state.saved_annotations_df)
            st.session_state.saved_annotations_df.loc[new_row_index] = {
                'text': st.session_state.current_response['context'],
                'class_of_interest': class_of_interest,
                'summary': st.session_state.current_response['summary'],
                'annotations': st.session_state.current_response['annotations'],
                'main_entities': st.session_state.current_response['main_entities'],
                'linked_annotations': st.session_state.current_response['linked_annotations']
            }

    st.divider()
    st.header("Saved Results")
    st.write(st.session_state.saved_annotations_df)

    st.divider()
    st.header("Comparison")
    st.write("Here you can compare the results of the pipeline with the saved results (once you have some saved results!)")

    compare_col_1, compare_col_2 = st.columns(2)
    with compare_col_1:
        if len(st.session_state.saved_annotations_df) > 0:
            comp_1_index = st.selectbox("Select the index of the saved results you want to compare with the pipeline results", st.session_state.saved_annotations_df.index, key = "comp_1_index")
            # If the index is selected, display the results
            if comp_1_index+1:
                # Display the results the same way as in the saved results section
                st.subheader("Annotated Text")
                formatted_annotations = format_annotations_for_display(st.session_state.saved_annotations_df.loc[comp_1_index]['text'], st.session_state.saved_annotations_df.loc[comp_1_index]['annotations'], st.session_state.saved_annotations_df.loc[comp_1_index]['class_of_interest'])
                annotated_text(*formatted_annotations)
                st.subheader("Pipeline Results")
                st.write("Summary")
                st.write(st.session_state.saved_annotations_df.loc[comp_1_index]['summary'])
                st.write("Annotations")
                st.write(st.session_state.saved_annotations_df.loc[comp_1_index]['annotations'])
                st.write("Main Entities")
                st.write(st.session_state.saved_annotations_df.loc[comp_1_index]['main_entities'])
                st.write("Linked Entities")
                st.write(st.session_state.saved_annotations_df.loc[comp_1_index]['linked_annotations'])
    with compare_col_2:
        if len(st.session_state.saved_annotations_df) > 0:
            comp_2_index = st.selectbox("Select the index of the saved results you want to compare with the pipeline results", st.session_state.saved_annotations_df.index, key = "comp_2_index")
            if comp_2_index:
                st.subheader("Annotated Text")
                formatted_annotations = format_annotations_for_display(st.session_state.saved_annotations_df.loc[comp_2_index]['text'], st.session_state.saved_annotations_df.loc[comp_2_index]['annotations'], st.session_state.saved_annotations_df.loc[comp_2_index]['class_of_interest'])
                annotated_text(*formatted_annotations)
                st.subheader("Pipeline Results")
                st.write("Summary")
                st.write(st.session_state.saved_annotations_df.loc[comp_2_index]['summary'])
                st.write("Annotations")
                st.write(st.session_state.saved_annotations_df.loc[comp_2_index]['annotations'])
                st.write("Main Entities")
                st.write(st.session_state.saved_annotations_df.loc[comp_2_index]['main_entities'])
                st.write("Linked Entities")
                st.write(st.session_state.saved_annotations_df.loc[comp_2_index]['linked_annotations'])

    # Footer
    st.markdown("---")
    st.markdown("Built for the Stelar project.") 
    st.markdown("---")
