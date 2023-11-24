import pandas as pd
import sys
import spacy
import torch
import flair
import stanza
from backend_functions import *

CONFIG_FILE = '../config_file.ini'
OUTPUT_FILE = 'output.json'

df_tools = make_df_info()
names_df = ["PERSON", "NORP", "FAC", "ORG","GPE", "LOC", "PRODUCT", "DATE","TIME","PERCENT","MONEY",\
        "QUANTITY","ORDINAL", "CARDINAL", "EVENT", "WORK_OF_ART", "LAW","LANGUAGE","MISC"]
df_categories_supported = make_df_categories_supported(names_df)

df = make_df_by_argument(sys.argv,CONFIG_FILE)

#Specify number of texts to be counted
N = choose_num(df,100)	#defaults to 100, if dataset has fewer, N = length of dataset

nlp_spacy_rob,nlp_spacy,tagger,nlp_stanza = load_import_models()		 		 #Load and import models

dictionary = {}
start,end,dictionary = find_named_entities(dictionary,N,nlp_spacy_rob,df,'spaCy + RoBERTa')	 #Find named entities of input file

if 'named_entities' in df.columns:										 #Ground truth provided, so perform evaluation
  df_results = evaluate_tool(N, names_df, nlp_spacy_rob, df) 	 				 #Evaluate tool on dataset
  dictionary = measure_write_result(df_results, dictionary, 'spaCy + RoBERTa', start, end) 	 #Print results after comparison with ground truth

start,end,dictionary = find_named_entities(dictionary,N,nlp_spacy,df,'spaCy')
if 'named_entities' in df.columns:										 
  df_results = evaluate_tool(N, names_df, nlp_spacy, df) 
  dictionary = measure_write_result(df_results, dictionary, 'spaCy', start, end)

start,end,dictionary = find_named_entities_flair(dictionary,N,tagger,df)
if 'named_entities' in df.columns:										
  df_results = evaluate_flair(N, names_df, tagger, df)
  dictionary = measure_write_result(df_results, dictionary, 'Flair', start, end)

start,end,dictionary = find_named_entities_stanza(dictionary,N,nlp_stanza,df)
if 'named_entities' in df.columns:										
  df_results = evaluate_stanza(N, names_df, nlp_stanza, df)
  dictionary = measure_write_result(df_results, dictionary, 'Stanza', start, end)

write_json_file(dictionary,OUTPUT_FILE,CONFIG_FILE)  #Write json file
