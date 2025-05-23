import streamlit as st
import numpy as np
import pandas as pd
from config import ANNOTATIONS_CSV_PATH, CANDIDATE_PAIRS_CSV_PATH

from annotated_text import annotated_text
import re
import ast

def go_to_index(index):
    """
    Updates the index and session state variables based on the given index.
    
    Parameters:
    index (int): The new index value.
    
    Returns:
    None
    """
    # change index
    if index: 
        index = int(index)
    st.session_state['df_index'] = index
    # read again the two dfs to have the latest results 
    st.session_state['matches_df'] = pd.read_csv(CANDIDATE_PAIRS_CSV_PATH)
    st.session_state['matches_df']['is_match'] = st.session_state['matches_df']['is_match']
    st.session_state['annotations_df'] = pd.read_csv(ANNOTATIONS_CSV_PATH)
    st.session_state['annotations_df']['foods_mistral'] = st.session_state['annotations_df']['foods_mistral'].apply(ast.literal_eval)
    st.session_state['annotations_df']['hazards_mistral'] = st.session_state['annotations_df']['hazards_mistral'].apply(ast.literal_eval)
    st.session_state['annotations_df']['organizations_spacy'] = st.session_state['annotations_df']['organizations_spacy'].apply(ast.literal_eval)
    st.session_state['annotations_df']['locations_spacy'] = st.session_state['annotations_df']['locations_spacy'].apply(ast.literal_eval)
    st.session_state['annotations_df']['dates_spacy'] = st.session_state['annotations_df']['dates_spacy'].apply(ast.literal_eval)
    # read is_annotated as int 
    st.session_state['annotations_df']['is_annotated'] = st.session_state['annotations_df']['is_annotated'].astype(bool)
    # identify the first product
    cur_id_1 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id1']
    cur_row_1 = st.session_state['annotations_df'].loc[st.session_state['annotations_df']['original id'] == cur_id_1]
    st.session_state['cur_product_1'] = cur_row_1.iloc[0]
    st.session_state['cur_product_options_1'] = st.session_state['cur_product_1']['foods_mistral']
    st.session_state['cur_product_options_1_selected'] = []
    st.session_state['cur_hazard_options_1'] = st.session_state['cur_product_1']['hazards_mistral']
    st.session_state['cur_hazard_options_1_selected'] = []    
    # identify the second product
    cur_id_2 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id2']
    cur_row_2 = st.session_state['annotations_df'].loc[st.session_state['annotations_df']['original id'] == cur_id_2]
    st.session_state['cur_product_2'] = cur_row_2.iloc[0]
    st.session_state['cur_product_options_2'] = st.session_state['cur_product_2']['foods_mistral']
    st.session_state['cur_product_options_2_selected'] = []
    st.session_state['cur_hazard_options_2'] = st.session_state['cur_product_2']['hazards_mistral']
    st.session_state['cur_hazard_options_2_selected'] = []

def save_annotation_1(products, hazards):
    """
    Saves the annotation for a specific row in the annotations CSV file.

    Parameters:
    - products (str): The updated products value.
    - hazards (str): The updated hazards value.

    Returns:
    None
    """
    # read again annotations csv to have the latest version 
    tmp_annot = pd.read_csv(ANNOTATIONS_CSV_PATH)
    # find the index of the current row
    cur_id = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id1']
    cur_row = tmp_annot.loc[tmp_annot['original id'] == cur_id]
    # update the products and hazards columns
    cur_row['products_ground'] = [products]
    cur_row['hazards_ground'] = [hazards]
    cur_row['is_annotated'] = [1]
    # save the updated row back to the annotations csv
    tmp_annot.loc[tmp_annot['original id'] == cur_id] = cur_row
    tmp_annot.to_csv(ANNOTATIONS_CSV_PATH, index=False)
    st.success("Saved successfully!")
    go_to_index(st.session_state['df_index'])

def save_annotation_2(products, hazards):
    """
    Saves the annotation for a specific row in the annotations CSV file.

    Parameters:
    - products (str): The updated products value.
    - hazards (str): The updated hazards value.

    Returns:
    None
    """
    # read again annotations csv to have the latest version 
    tmp_annot = pd.read_csv(ANNOTATIONS_CSV_PATH)
    # find the index of the current row
    cur_id = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id2']
    cur_row = tmp_annot.loc[tmp_annot['original id'] == cur_id]
    # update the products and hazards columns
    cur_row['products_ground'] = [products]
    cur_row['hazards_ground'] = [hazards]
    cur_row['is_annotated'] = [1]
    # save the updated row back to the annotations csv
    tmp_annot.loc[tmp_annot['original id'] == cur_id] = cur_row
    tmp_annot.to_csv(ANNOTATIONS_CSV_PATH, index=False)
    st.success("Saved successfully!")
    go_to_index(st.session_state['df_index'])


def save_match(is_match, is_related):
    """
    Saves the match information for a candidate pair.

    Args:
        is_match (str): Indicates whether the candidate pair is a match or not. 
                        Should be either "Yes" or "No".
        is_related (str): Indicates whether the candidate pair is related or not. 
                          Should be either "Yes" or "No".

    Returns:
        None
    """
    # read the candidate pair csv 
    tmp_matches = pd.read_csv(CANDIDATE_PAIRS_CSV_PATH)
    # find the index of the current row
    cur_match = tmp_matches.iloc[st.session_state['df_index']]
    # update the is_match column
    if is_match == "Yes":
        cur_match['is_match'] = 1
    else:
        cur_match['is_match'] = 0
    if is_related == "Yes":
        cur_match['is_related'] = 1
    else:
        cur_match['is_related'] = 0
    # save the updated row back to the candidate pairs csv
    tmp_matches.iloc[st.session_state['df_index']] = cur_match
    tmp_matches.to_csv(CANDIDATE_PAIRS_CSV_PATH, index=False)
    st.success("Saved successfully!")
    go_to_index(st.session_state['df_index'])


def save_annotation(products_1, hazards_1, products_2, hazards_2, is_match):
    """
    Saves the annotation and match result to the corresponding CSV files.

    Args:
        products_1 (str): The products for the first item.
        hazards_1 (str): The hazards for the first item.
        products_2 (str): The products for the second item.
        hazards_2 (str): The hazards for the second item.
        is_match (str): The match result ("Yes" or "No").

    Returns:
        None
    """
    # read again annotations csv to have the latest version 
    tmp_annot = pd.read_csv(ANNOTATIONS_CSV_PATH)
    # find the index of the current row
    cur_id_1 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id1']
    cur_row_1 = tmp_annot.loc[tmp_annot['original id'] == cur_id_1]
    cur_id_2 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id2']
    cur_row_2 = tmp_annot.loc[tmp_annot['original id'] == cur_id_2]
    # update the products and hazards columns
    cur_row_1['products_ground'] = [products_1]
    cur_row_1['hazards_ground'] = [hazards_1]
    cur_row_2['products_ground'] = [products_2]
    cur_row_2['hazards_ground'] = [hazards_2]
    cur_row_1['is_annotated'] = [1]
    cur_row_2['is_annotated'] = [1]
    # save the updated rows back to the annotations csv
    tmp_annot.loc[tmp_annot['original id'] == cur_id_1] = cur_row_1
    tmp_annot.loc[tmp_annot['original id'] == cur_id_2] = cur_row_2
    tmp_annot.to_csv(ANNOTATIONS_CSV_PATH, index=False)
    # save the match result
    tmp_matches = pd.read_csv(CANDIDATE_PAIRS_CSV_PATH)
    cur_match = tmp_matches.iloc[st.session_state['df_index']]
    if is_match == "Yes":
        cur_match['is_match'] = 1
    else:
        cur_match['is_match'] = 0
    tmp_matches.iloc[st.session_state['df_index']] = cur_match
    tmp_matches.to_csv(CANDIDATE_PAIRS_CSV_PATH, index=False)
    st.success("Saved successfully!")


def next_button_clicked():  
    """
    Function to handle the click event of the next button.
    This function updates the index, reads the latest dataframes, and sets the current product options and selections.
    """
    # change index
    st.session_state['df_index'] += 1 
    # read again the two dfs to have the latest results 
    st.session_state['matches_df'] = pd.read_csv(CANDIDATE_PAIRS_CSV_PATH)
    st.session_state['matches_df']['is_match'] = st.session_state['matches_df']['is_match']
    st.session_state['matches_df']['is_related'] = st.session_state['matches_df']['is_related']
    st.session_state['annotations_df'] = pd.read_csv(ANNOTATIONS_CSV_PATH)
    st.session_state['annotations_df']['foods_mistral'] = st.session_state['annotations_df']['foods_mistral'].apply(ast.literal_eval)
    st.session_state['annotations_df']['hazards_mistral'] = st.session_state['annotations_df']['hazards_mistral'].apply(ast.literal_eval)
    st.session_state['annotations_df']['organizations_spacy'] = st.session_state['annotations_df']['organizations_spacy'].apply(ast.literal_eval)
    st.session_state['annotations_df']['locations_spacy'] = st.session_state['annotations_df']['locations_spacy'].apply(ast.literal_eval)
    st.session_state['annotations_df']['dates_spacy'] = st.session_state['annotations_df']['dates_spacy'].apply(ast.literal_eval)
    # read is_annotated as bool 
    st.session_state['annotations_df']['is_annotated'] = st.session_state['annotations_df']['is_annotated'].astype(bool)  
    # identify the first product
    cur_id_1 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id1']
    cur_row_1 = st.session_state['annotations_df'].loc[st.session_state['annotations_df']['original id'] == cur_id_1]
    st.session_state['cur_product_1'] = cur_row_1.iloc[0]
    st.session_state['cur_product_options_1'] = st.session_state['cur_product_1']['foods_mistral']
    st.session_state['cur_hazard_options_1'] = st.session_state['cur_product_1']['hazards_mistral']
    # if it is annotated show the existing annotations otherwise empty 
    st.session_state['cur_product_options_1_selected'] = []
    st.session_state['cur_hazard_options_1_selected'] = []
    # identify the second product
    cur_id_2 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id2']
    cur_row_2 = st.session_state['annotations_df'].loc[st.session_state['annotations_df']['original id'] == cur_id_2]
    st.session_state['cur_product_2'] = cur_row_2.iloc[0]
    st.session_state['cur_product_options_2'] = st.session_state['cur_product_2']['foods_mistral']
    st.session_state['cur_hazard_options_2'] = st.session_state['cur_product_2']['hazards_mistral']
    st.session_state['cur_product_options_2_selected'] = []
    st.session_state['cur_hazard_options_2_selected'] = []


def prev_button_clicked():
    """
    Function to handle the click event of the previous button.
    It updates the index, reads the necessary dataframes, and updates the session state variables accordingly.
    """
    # change index
    if st.session_state['df_index'] > 0:
        st.session_state['df_index'] += -1 
        # identify the first product
        # read again the two dfs to have the latest results 
        st.session_state['matches_df'] = pd.read_csv(CANDIDATE_PAIRS_CSV_PATH)
        st.session_state['matches_df']['is_match'] = st.session_state['matches_df']['is_match']
        st.session_state['matches_df']['is_related'] = st.session_state['matches_df']['is_related']
        st.session_state['annotations_df'] = pd.read_csv(ANNOTATIONS_CSV_PATH)
        st.session_state['annotations_df']['foods_mistral'] = st.session_state['annotations_df']['foods_mistral'].apply(ast.literal_eval)
        st.session_state['annotations_df']['hazards_mistral'] = st.session_state['annotations_df']['hazards_mistral'].apply(ast.literal_eval)
        st.session_state['annotations_df']['organizations_spacy'] = st.session_state['annotations_df']['organizations_spacy'].apply(ast.literal_eval)
        st.session_state['annotations_df']['locations_spacy'] = st.session_state['annotations_df']['locations_spacy'].apply(ast.literal_eval)
        st.session_state['annotations_df']['dates_spacy'] = st.session_state['annotations_df']['dates_spacy'].apply(ast.literal_eval)
        # read is_annotated as int 
        st.session_state['annotations_df']['is_annotated'] = st.session_state['annotations_df']['is_annotated'].astype(bool)          
        cur_id_1 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id1']
        cur_row_1 = st.session_state['annotations_df'].loc[st.session_state['annotations_df']['original id'] == cur_id_1]
        st.session_state['cur_product_1'] = cur_row_1.iloc[0]
        st.session_state['cur_product_options_1'] = st.session_state['cur_product_1']['foods_mistral']
        st.session_state['cur_product_options_1_selected'] = []
        st.session_state['cur_hazard_options_1'] = st.session_state['cur_product_1']['hazards_mistral']
        st.session_state['cur_hazard_options_1_selected'] = []
        # identify the second product
        cur_id_2 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id2']
        cur_row_2 = st.session_state['annotations_df'].loc[st.session_state['annotations_df']['original id'] == cur_id_2]
        st.session_state['cur_product_2'] = cur_row_2.iloc[0]
        st.session_state['cur_product_options_2'] = st.session_state['cur_product_2']['foods_mistral']
        st.session_state['cur_product_options_2_selected'] = []
        st.session_state['cur_hazard_options_2'] = st.session_state['cur_product_2']['hazards_mistral']
        st.session_state['cur_hazard_options_2_selected'] = []




# add index as an st.state variable 
if "df_index" not in st.session_state:
    st.session_state['df_index'] = 0
if "matches_df" not in st.session_state:
    st.session_state['matches_df'] = pd.read_csv(CANDIDATE_PAIRS_CSV_PATH)
    st.session_state['matches_df']['is_match'] = st.session_state['matches_df']['is_match']
    st.session_state['matches_df']['is_related'] = st.session_state['matches_df']['is_related']
if "annotations_df" not in st.session_state:
    st.session_state['annotations_df'] = pd.read_csv(ANNOTATIONS_CSV_PATH)
    st.session_state['annotations_df']['foods_mistral'] = st.session_state['annotations_df']['foods_mistral'].apply(ast.literal_eval)
    st.session_state['annotations_df']['hazards_mistral'] = st.session_state['annotations_df']['hazards_mistral'].apply(ast.literal_eval)
    st.session_state['annotations_df']['organizations_spacy'] = st.session_state['annotations_df']['organizations_spacy'].apply(ast.literal_eval)
    st.session_state['annotations_df']['locations_spacy'] = st.session_state['annotations_df']['locations_spacy'].apply(ast.literal_eval)
    st.session_state['annotations_df']['dates_spacy'] = st.session_state['annotations_df']['dates_spacy'].apply(ast.literal_eval)
    # read is_annotated as int 
    st.session_state['annotations_df']['is_annotated'] = st.session_state['annotations_df']['is_annotated'].astype(bool)


if "cur_product_1" not in st.session_state:
    cur_id_1 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id1']
    cur_row_1 = st.session_state['annotations_df'].loc[st.session_state['annotations_df']['original id'] == cur_id_1]
    st.session_state['cur_product_1'] = cur_row_1.iloc[0]
if "cur_product_options_1" not in st.session_state:
    st.session_state['cur_product_options_1'] = st.session_state['cur_product_1']['foods_mistral']
if "cur_product_options_1_selected" not in st.session_state:
    st.session_state['cur_product_options_1_selected'] = []


if "cur_hazard_options_1" not in st.session_state:
    st.session_state['cur_hazard_options_1'] = st.session_state['cur_product_1']['hazards_mistral']
if "cur_hazard_options_1_selected" not in st.session_state:
    st.session_state['cur_hazard_options_1_selected'] = []

# repear the same for the second product
if "cur_product_2" not in st.session_state:
    cur_id_2 = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id2']
    cur_row_2 = st.session_state['annotations_df'].loc[st.session_state['annotations_df']['original id'] == cur_id_2]
    st.session_state['cur_product_2'] = cur_row_2.iloc[0]
if "cur_product_options_2" not in st.session_state:
    st.session_state['cur_product_options_2'] = st.session_state['cur_product_2']['foods_mistral']
if "cur_product_options_2_selected" not in st.session_state:
    st.session_state['cur_product_options_2_selected'] = []

if "cur_hazard_options_2" not in st.session_state:
    st.session_state['cur_hazard_options_2'] = st.session_state['cur_product_2']['hazards_mistral']
if "cur_hazard_options_2_selected" not in st.session_state:
    st.session_state['cur_hazard_options_2_selected'] = []


def add_other_product_1(other_product):
    """
    Adds a new product to the list of options for the current product 1.

    Parameters:
    - other_product (str): The name of the product to be added.

    Returns:
    None
    """
    if other_product.lower() in st.session_state['cur_product_1']['description'].lower():
        st.session_state['cur_product_options_1'].append(other_product)
        st.write(f"New product added: {other_product}")
    else:
        st.write(f"Product {other_product} not found in the description!")

def add_other_hazard_1(other_hazard):
    """
    Adds a new hazard to the hazard options for the current product.

    Parameters:
    other_hazard (str): The hazard to be added.

    Returns:
    None
    """
    if other_hazard.lower() in st.session_state['cur_product_1']['description'].lower():
        st.session_state['cur_hazard_options_1'].append(other_hazard)
        st.write(f"New hazard added: {other_hazard}")
    else:
        st.write(f"Hazard {other_hazard} not found in the description!")

def add_other_product_2(other_product):
    """
    Adds a new product to the current product options list.

    Parameters:
        other_product (str): The name of the product to be added.

    Returns:
        None
    """
    if other_product.lower() in st.session_state['cur_product_2']['description'].lower():
        st.session_state['cur_product_options_2'].append(other_product)
        st.write(f"New product added: {other_product}")
    else:
        st.write(f"Product {other_product} not found in the description!")

def add_other_hazard_2(other_hazard):
    """
    Adds a new hazard to the current hazard options for the second product.

    Parameters:
    other_hazard (str): The hazard to be added.

    Returns:
    None
    """
    if other_hazard.lower() in st.session_state['cur_product_2']['description'].lower():
        st.session_state['cur_hazard_options_2'].append(other_hazard)
        st.write(f"New hazard added: {other_hazard}")
    else:
        st.write(f"Hazard {other_hazard} not found in the description!")

def annotate_preprocess(df_row):
    """
    Preprocesses the annotations in a DataFrame row and returns the annotations and labels.

    Args:
        df_row (pandas.Series): A row from a DataFrame containing the following columns:
                                - 'foods_mistral': List of food annotations
                                - 'hazards_mistral': List of hazard annotations
                                - 'organizations_spacy': List of organization annotations
                                - 'locations_spacy': List of location annotations
                                - 'dates_spacy': List of date annotations

    Returns:
        tuple: A tuple containing two lists:
               - annotations: A list of all the annotations concatenated together
               - labels: A list of labels corresponding to each annotation
    """
    products = df_row['foods_mistral']
    hazards = df_row['hazards_mistral']
    organizations = df_row['organizations_spacy']
    locations = df_row['locations_spacy']
    dates = df_row['dates_spacy']

    # Concatenate all the annotations into one list
    annotations = products + hazards + organizations + locations + dates

    # Create a list of labels the same length as the annotations
    labels = ['product'] * len(products) + ['hazard'] * len(hazards) + ['organization'] * len(organizations) + ['location'] * len(locations) + ['date'] * len(dates)

    return annotations, labels


def annotate_keyword(text, keywords, labels):
    """
    Annotates keywords in a given text with corresponding labels.

    Args:
        text (str): The text to be annotated.
        keywords (list): A list of keywords to be annotated.
        labels (list): A list of labels corresponding to the keywords.

    Returns:
        str: The annotated text.

    Raises:
        ValueError: If the lengths of `keywords` and `labels` are not the same.
    """

    if len(keywords) != len(labels):
        raise ValueError("keywords and labels must have the same length")

    replacements = []  # List to track replacements
    marked_positions = []  # List to keep track of marked positions

    for keyword, label in zip(keywords, labels):
        if type(keyword) is not str:
            continue

        start_pos = 0
        while True:
            found_pos = text.lower().find(keyword.lower(), start_pos)
            if found_pos == -1:
                break

            end_pos = found_pos + len(keyword)

            if any(start <= found_pos < end for start, end in marked_positions):
                start_pos = end_pos
                continue

            marked_positions.append((found_pos, end_pos))
            # Adjusted placeholder to include spaces before '{' and after '}'
            placeholder = f' {{ {keyword} }} '
            replacements.append((found_pos, end_pos, placeholder))
            start_pos = end_pos

    # Sort replacements by start position in reverse order
    replacements.sort(key=lambda x: x[0], reverse=True)

    # Apply replacements in reverse order
    for start_pos, end_pos, placeholder in replacements:
        text = text[:start_pos] + placeholder + text[end_pos:]

    phrases_and_spaces = re.findall(r'\{.*?\}|\S+|\s+', text)

    annotated_text_list = []
    for phrase_or_space in phrases_and_spaces:
        if phrase_or_space.startswith('{') and phrase_or_space.endswith('}'):
            keyword = phrase_or_space[1:-1].strip()
            if keyword in keywords:
                index = keywords.index(keyword)
                annotated_text_list.append((keyword, labels[index]))
            else:
                annotated_text_list.append(phrase_or_space)
        else:
            annotated_text_list.append(phrase_or_space)

    # Display the annotated text (assuming annotated_text is defined elsewhere)
    annotated_text(*annotated_text_list)




st.set_page_config(
    page_title="Linking&NER Evaluate",
    page_icon="🧊",
    layout="wide",
)



st.title("Linking&NER Evaluate")
st.subheader("A dashboard to evaluate and annotate results for Named Entity Recognition and Linking.")

# Progress 
st.subheader("Progress")
# Show the progress of the annotation by showing the number of is_annottated == 1 over the total len of the annotations df

st.write(f"Progress: {st.session_state['annotations_df']['is_annotated'].sum()}/{len(st.session_state['annotations_df'])}")
# Calculate the number of cases where 'is_match' is not null
num_not_null_is_match = st.session_state['matches_df']['is_match'].notna().sum()
# Calculate the total number of cases
total_cases = len(st.session_state['matches_df'])


# Display the match annotations progress
st.write(f"Match Annotations Progress: {num_not_null_is_match}/{total_cases} ({(num_not_null_is_match/total_cases)*100:.2f}%)")

# Show the progress in terms of match annotation 

st.write("Move to case:")
# add a int input to move to a specific case
case_to_move_to = st.number_input("Case to move to", min_value=0, max_value=len(st.session_state['matches_df'])-1, value=0, step=1)
# add a button to move to the case
move_button = st.button("Move to case", on_click=go_to_index, args=(case_to_move_to,))
# find the first index in the matches_df where is_match is False
first_false_index = st.session_state['matches_df'][st.session_state['matches_df']['is_match'].isnull()].index.min()
move_to_last_button = st.button("Move to first not annotated case.", on_click=go_to_index, args=(first_false_index,))
# show the annotations_df 
#st.write(st.session_state['matches_df'])

# show just the text from the first row of annotations df 
#st.write(st.session_state['annotations_df'].iloc[st.session_state['df_index']]['foods_mistral'])

st.subheader("Current Case")
# show the row of the matches_df that corresponds to the current index
st.write(st.session_state['matches_df'].iloc[st.session_state['df_index']])

cur_id = st.session_state['matches_df'].iloc[st.session_state['df_index']]['Original id1']
tmp_row = st.session_state['annotations_df'].loc[st.session_state['annotations_df']['original id'] == cur_id]
# select only the first row from tmp row
tmp_row = tmp_row.iloc[0]
tmp_annotations, tmp_labels = annotate_preprocess(tmp_row)

#annotate_keyword(st.session_state['matches_df'].iloc[st.session_state['df_index']]['description_1'], tmp_annotations, tmp_labels)

#annotated_text(st.session_state['annotations_df'].iloc[st.session_state['df_index']]['text'], st.session_state['annotations_df'].iloc[st.session_state['df_index']]['foods_mistral'], 'foods')


col1, col2 = st.columns(2, gap="large")
with col1:
    container_1 = st.container(border=True)
    with container_1:
        st.subheader("Incident 1")
        st.write(st.session_state['cur_product_1'])
        enable_annot_1 = st.checkbox("Enable Annotations", key="enable_annot_1")
        st.subheader("Title")
        if enable_annot_1:
            annotations_1, labels_1 = annotate_preprocess(st.session_state['cur_product_1'])
            annotate_keyword(st.session_state['cur_product_1']['originalTitle'], annotations_1, labels_1)
        else:
            st.write(st.session_state['cur_product_1']['originalTitle'])
        st.subheader("Description")
        # have an enable annotations checkbox 
        # if active show annotated text else just write the text 
        
        if enable_annot_1:
            annotate_keyword(st.session_state['cur_product_1']['description'], annotations_1, labels_1)
        else:
            st.write(st.session_state['cur_product_1']['description'])
        
        if st.session_state['cur_product_1']['is_annotated']:
            st.write(f":green[Already annotated!]")
            st.write(f":blue[Products]: {st.session_state['cur_product_1']['products_ground']}")
            st.write(f":blue[Hazards] {st.session_state['cur_product_1']['hazards_ground']}")
        else:
            st.write(f":red[Not annotated yet!]")
        st.write("Products")
        st.write("From the following products choose all that are present in the incident:")
        for i, product in enumerate(st.session_state['cur_product_options_1']):
            if st.checkbox(product, key=f'prod_1_{i}'):
                if product not in st.session_state['cur_product_options_1_selected']:
                    st.session_state['cur_product_options_1_selected'].append(product)
            else:
                if product in st.session_state['cur_product_options_1_selected']:
                    st.session_state['cur_product_options_1_selected'].remove(product)
        st.write(st.session_state['cur_product_options_1_selected'])
        # add a textbox where I can add a new organization
        other_product_1 = st.text_input('Other product', key='other_product_1')
        # add a button that calls the add_new_org function
        st.button('Add other product',key="add_prod_1", on_click=add_other_product_1, args=(other_product_1,))
        st.write("Hazards")
        st.write("From the following hazards choose all that are present in the incident:")
        for i, hazard in enumerate(st.session_state['cur_hazard_options_1']):
            if st.checkbox(hazard, key=f'hazard_1_{i}'):
                if hazard not in st.session_state['cur_hazard_options_1_selected']:
                    st.session_state['cur_hazard_options_1_selected'].append(hazard)
            else:
                if hazard in st.session_state['cur_hazard_options_1_selected']:
                    st.session_state['cur_hazard_options_1_selected'].remove(hazard)
        st.write(st.session_state['cur_hazard_options_1_selected'])
        # add a textbox where I can add a new organization
        other_hazard_1 = st.text_input('Other hazard', key='other_hazard_1')
        # add a button that calls the add_new_org function
        st.button('Add other hazard',key="add_haz_1", on_click=add_other_hazard_1, args=(other_hazard_1,))
        st.write("Save")
        st.button("Save Annotations", on_click=save_annotation_1, args=(st.session_state['cur_product_options_1_selected'], st.session_state['cur_hazard_options_1_selected'],), key="save_annot_1")
with col2:
    container_2 = st.container(border=True)
    with container_2:
        st.subheader("Incident 2")
        # repeat the same for the second product
        st.write(st.session_state['cur_product_2'])
        enable_annot_2 = st.checkbox("Enable Annotations", key="enable_annot_2")
        st.subheader("Title")
        if enable_annot_2:
            annotations_2, labels_2 = annotate_preprocess(st.session_state['cur_product_2'])
            annotate_keyword(st.session_state['cur_product_2']['originalTitle'], annotations_2, labels_2)
        else:
            st.write(st.session_state['cur_product_2']['originalTitle'])
        st.subheader("Description")
        if enable_annot_2:
            annotate_keyword(st.session_state['cur_product_2']['description'], annotations_2, labels_2)
        else:
            st.write(st.session_state['cur_product_2']['description'])
        # add one checkbox for each value in st.session_state['cur_products']
        # if is_annotated == True show a green indicator that the product is already annotated 
        # and print the annotations 
        if st.session_state['cur_product_2']['is_annotated']:
            st.write(f":green[Already annotated!]")
            st.write(f":blue[Products]: {st.session_state['cur_product_2']['products_ground']}")
            st.write(f":blue[Hazards] {st.session_state['cur_product_2']['hazards_ground']}")
        else:
            st.write(f":red[Not annotated yet!]")
        st.write("Products")
        st.write("From the following products choose all that are present in the incident:")
        for i, product in enumerate(st.session_state['cur_product_options_2']):
            if st.checkbox(product, key=f'prod_2_{i}'):
                if product not in st.session_state['cur_product_options_2_selected']:
                    st.session_state['cur_product_options_2_selected'].append(product)
            else:
                if product in st.session_state['cur_product_options_2_selected']:
                    st.session_state['cur_product_options_2_selected'].remove(product)
        st.write(st.session_state['cur_product_options_2_selected'])
        # add a textbox where I can add a new organization
        other_product_2 = st.text_input('Other product', key='other_product_2')
        # add a button that calls the add_new_org function
        st.button('Add other product',key="add_prod_2", on_click=add_other_product_2, args=(other_product_2,))
        st.write("Hazards")
        st.write("From the following hazards choose all that are present in the incident:")
        for i, hazard in enumerate(st.session_state['cur_hazard_options_2']):
            if st.checkbox(hazard, key=f'hazard_2_{i}'):
                if hazard not in st.session_state['cur_hazard_options_2_selected']:
                    st.session_state['cur_hazard_options_2_selected'].append(hazard)
            else:
                if hazard in st.session_state['cur_hazard_options_2_selected']:
                    st.session_state['cur_hazard_options_2_selected'].remove(hazard)
        st.write(st.session_state['cur_hazard_options_2_selected'])
        # add a textbox where I can add a new organization
        other_hazard_2 = st.text_input('Other hazard', key='other_hazard_2')
        # add a button that calls the add_new_org function
        st.button('Add other hazard',key="add_haz_2", on_click=add_other_hazard_2, args=(other_hazard_2,))
        st.write("Save")
        st.button("Save Annotations", on_click=save_annotation_2, args=(st.session_state['cur_product_options_2_selected'], st.session_state['cur_hazard_options_2_selected'],), key = "save_annot_2")


st.subheader("Evaluation")
# if is_match is not None write Already annotatetd and show the value 
# if is_match is None show a radio button to select if it is the same incident or not


# Assuming 'is_match' is a column in 'matches_df' DataFrame and you're accessing the current row's value
current_is_match = st.session_state['matches_df'].iloc[st.session_state['df_index']]['is_match']
current_is_related = st.session_state['matches_df'].iloc[st.session_state['df_index']]['is_related']

if pd.isna(current_is_match):
    # 'is_match' is None, show a radio button to select if it is the same incident or not
    st.write(f":red[Not annotated yet!]")
else:
    # 'is_match' is not None, display its status
    st.write(f":green[Already annotated!]")
    if current_is_match:
        st.write(f":blue[It's a match! They refer to the same incident!]")
    else:
        st.write(f":blue[It's not a match! They refer to different incidents!]")

if not pd.isna(current_is_related):
    if current_is_related:
        st.write(f":blue[They are closely related!]")
    else:
        st.write(f":blue[They are not closely related!]")
        
same_incident = st.radio("Is it the same incident?", ["Yes", "No"])
closely_related_incident = st.radio("Is it a closely related incident?", ["Yes", "No"])
# add a button to save the annotation
save_match_button = st.button("Save Match", on_click=save_match, args=(same_incident,closely_related_incident,))

# Add a previous button
prev_button = st.button("Previous", on_click=prev_button_clicked)
# When prev_button is clicked decrement index 
next_button = st.button("Next", on_click=next_button_clicked)

