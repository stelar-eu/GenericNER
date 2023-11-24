import pandas as pd
import re
import time
import json
import csv
import spacy
import en_core_web_trf
import en_core_web_sm
import torch
import flair
import stanza
from datasets import load_dataset
from flair.nn import Classifier
from flair.data import Sentence
from configparser import ConfigParser

def df_add_value(df, idx, col, value):
  df.loc[idx,col] = value
  return df

def make_df_info():
  df_tools = pd.DataFrame(index = ['spaCy', 'spaCy + RoBERTa benchmark', 'Flair', 'StanfordNLP'], columns = ['Capabilities','Execution Time','Avg. Hit Rate'])

  df_tools = df_add_value(df_tools, 'spaCy','Execution Time', '10 s')
  df_tools = df_add_value(df_tools, 'spaCy + RoBERTa benchmark','Execution Time', '10 s')
  df_tools = df_add_value(df_tools, 'Flair','Execution Time', '10 m')
  df_tools = df_add_value(df_tools, 'StanfordNLP','Execution Time', '6 m')

  df_tools = df_add_value(df_tools, 'spaCy','Avg. Hit Rate', '90.83%')
  df_tools = df_add_value(df_tools, 'spaCy + RoBERTa benchmark','Avg. Hit Rate', '90.83%')
  df_tools = df_add_value(df_tools, 'Flair','Avg. Hit Rate', '73.09%')
  df_tools = df_add_value(df_tools, 'StanfordNLP','Avg. Hit Rate', '96.18%')

  df_tools = df_add_value(df_tools, 'spaCy','Capabilities', 'All')
  df_tools = df_add_value(df_tools, 'spaCy + RoBERTa benchmark','Capabilities', 'All')
  df_tools = df_add_value(df_tools, 'Flair','Capabilities', 'PER,LOC,ORG')
  df_tools = df_add_value(df_tools, 'StanfordNLP','Capabilities', 'All')

  return df_tools

def make_df_categories_supported(names_df):
  df_categories_supported = pd.DataFrame(index = ['spaCy', 'spaCy + RoBERTa benchmark', 'Flair', 'StanfordNLP'], columns = names_df)
  for name in names_df:
    df_categories_supported = df_add_value(df_categories_supported, 'spaCy', name, 'Yes')
    df_categories_supported = df_add_value(df_categories_supported, 'spaCy + RoBERTa benchmark', name, 'Yes')
    df_categories_supported = df_add_value(df_categories_supported, 'StanfordNLP', name, 'Yes')
    if name in ['PERSON','LOC','ORG','GPE']:
      df_categories_supported = df_add_value(df_categories_supported, 'Flair', name, 'Yes')
    else:
      df_categories_supported = df_add_value(df_categories_supported, 'Flair', name, 'No')
  return df_categories_supported
  

def prepare_default_dataset(dataset):
  """
  Builds dataframe that stores texts, from 'dataset' argument. 
  
  Args:
      dataset (class Dataset from datasets library): A dataset from HuggingFace. Its texts will be used for entity labelling.
  
  Returns:
      The dataframe built
  """
  names=["O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC", "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC", "B-PRODUCT", "I-PRODUCT", "B-DATE", "I-DATE", "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT", "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY", "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL", "B-EVENT", "I-EVENT", "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE"]
  data_train = dataset['train']
  df = pd.DataFrame.from_dict(data_train[0])
  entities = []
  named_entities = []
  for i in range(len(df)):
    entities.append(df['sentences'][i]['named_entities'])
    df['sentences'][i] = df['sentences'][i]['words']
  for ent in entities:
    named_ent = []
    for element in ent:
      named_element = names[element]
      named_ent.append(named_element)
    named_entities.append(named_ent)
  df['named_entities'] = named_entities
  sentences_non_tokenized = []
  for i in range(len(df)):
    sentence = ""
    for word in df['sentences'][i]:
      sentence = sentence + ' ' + word
    sentences_non_tokenized.append(sentence)
  df['sentence_non_tokenized'] = sentences_non_tokenized
  return df

def read_config_file(config_file):
  """
  Reads configuration file. 
  
  Args:
      config_file (str): The configuration file to be read.
  
  Returns:
      Configuration file data
  """
  config = ConfigParser()
  config.read(config_file)
  config_data = config['DEFAULT']
  col_name_config = config_data['namecolumn']
  col_name_optional_config = config_data['namecolumn_optional']
  csv_delimiter = config_data['csv_delimiter']
  csv_delimiter = csv_delimiter[1:-1]
  output_path = config_data['output_file_path']
  return col_name_config,col_name_optional_config,csv_delimiter,output_path 

def check_input_with_config(conf_file,list_arguments):
  """
  Checks if input CSV file has the correct format, based on configuration file.
  
  Args:
      conf_file (str): The configuration file to be read.
      list_arguments (list): Arguments passed when running the progam.
  Returns:
      Configuration file data and dataframe containing CSV file data
  """
  col_name_config,col_name_optional_config,csv_delimiter,output_path  = read_config_file(conf_file)	#check if given csv in correct form
  df_input = pd.DataFrame()
  if len(list_arguments) == 1:
    print('ERROR:Please provide an input CSV file.')
  else:
    if '.csv' not in list_arguments[1]:
     df_input == pd.DataFrame()
    df_input = pd.read_csv(list_arguments[1], delimiter = csv_delimiter)
  return df_input,col_name_config,col_name_optional_config

def make_quick_csv():
  row_list = [["sentences","tags"],
              ["On August 17 , Taiwan 's investigation department and police solved the case and announced the March 19 shooting case was closed .",
                ['O','B-DATE','I-DATE','O','B-LOC','I-LOC','O','O','O','O','O','O','O','O','O','O','B-DATE','I-DATE','O','O','O','O','O']]]
  with open('input.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(row_list)

def prepare_given_dataset(df_input, col_name_config,col_name_optional_config):
  """
  Builds dataframe that stores texts, from 'df_input' argument.
  
  Args:
      df_input (pd.DataFrame): Dataframe containing data from input CSV file.
      col_name_config (str): CSV first column name, taken from config file
      col_name_optional_config (str):  CSV second column name, taken from config file
  
  Returns:
      The dataframe built
  """
  df = pd.DataFrame()
  if len(df_input.columns) == 2 and (df_input.columns[0] != col_name_config or df_input.columns[1] != col_name_optional_config):
    print('Wrong csv format. Please provide a csv file which contains a single column named \"', col_name_config, '" and optionally, a ground truth column named \"', col_name_optional_config, "\"")
  elif len(df_input.columns) == 1 and df_input.columns != col_name_config:
    print('Wrong csv format. Please provide a csv file which contains a single column named \"', col_name_config, '" and optionally, a ground truth column named \"', col_name_optional_config, "\"")
  elif len(df_input.columns) > 2:
    print('Wrong csv format. Please provide a csv file which contains a single column named \"', col_name_config, '" and optionally, a ground truth column named \"', col_name_optional_config, "\"")
  else:
    df = pd.DataFrame(index = range(len(df_input)), columns = ['sentences','sentence_non_tokenized'])
    for i in range(len(df_input)):
      df.loc[i,'sentences'] = df_input.loc[i,col_name_config].split()
    df['sentence_non_tokenized'] = df_input[col_name_config]
    df['sentence_non_tokenized'] = df['sentence_non_tokenized'].apply(lambda x: ' ' + x)
    if len(df_input.columns) == 2:
      df['named_entities'] = df_input[col_name_optional_config]
      df['named_entities'] = df['named_entities'].apply(lambda x: x[2:-2].split('\',\''))
  return df

def make_df_by_argument(list_arguments,config_file):
  """
  Returns dataframe according to whether a CSV file was given as an input or not.
  
  Args:
      list_arguments (list): List of arguments
      config_file (str): Configuration file
  
  Returns:
      The dataframe
  """
  if len(list_arguments) == 2 and list_arguments[1] == 'default':	#Ontonotes dataset
    ds = load_dataset("conll2012_ontonotesv5","english_v4")
    df = prepare_default_dataset(ds)
  elif len(list_arguments) == 1 or (len(list_arguments) == 2 and list_arguments[1] != 'default'):	#Dataset given from user
    df_input,col_name_config,col_name_optional_config  = check_input_with_config(config_file,list_arguments)
    if df_input.empty:
      print('ERROR: Please provide a .csv file.')
      exit()
    df = prepare_given_dataset(df_input, col_name_config,col_name_optional_config) 
    if df.empty:
      print('ERROR: Wrong CSV format!')
      exit()
  else:
    print('ERROR: Provide a single CSV file as an argument. Run example: python entity_extraction.py default or python entity_extraction.py your_file.csv')
    exit(1)
  return df

def choose_num(df, default):
  """
  How many texts will have their entities labelled. Defaults to 'default'. If 'default' is bigger than the number of texts, the whole dataset will be used for entity extraction.
  
  Args:
      df (pd.DataFrame): Dataframe containing texts
      default (int): Number of texts to be used for entity labelling
  
  Returns:
      Number chosen
  """
  N = default
  if N > len(df):
    N = len(df)
  return N

def load_import_models():
  """
  Load entity extraction models
  
  Returns:
      Models loaded
  """  
  #Load and import model
  nlp_spacy_rob = spacy.load("en_core_web_trf")
  nlp_spacy_rob = en_core_web_trf.load()
  #Import spaCy
  nlp_spacy = spacy.load("en_core_web_sm")
  nlp_spacy = en_core_web_sm.load()
  #Import flair, load model
  tagger = Classifier.load('ner-ontonotes')
  #Download and import pipeline
  stanza.download('en')
  nlp_stanza = stanza.Pipeline('en') # initialize English neural pipeline
  return nlp_spacy_rob,nlp_spacy,tagger,nlp_stanza

def add_result_to_df(index,df_results, predicted_entities, real_entities):
  """
  Compares entity extraction results to ground truth for a specific text, adds information in 'df_results'.
  
  Args:
      index(int): Text index
      df_results (pd.DataFrame): Dataframe storing evaluation results
      predicted_entities (list): Entity extraction results of a tool (e.g. spaCy or Flair)
      real_entities (list): Real entity labels
  Returns:
      df_results updated
  """
  
  if (len(predicted_entities) != len(real_entities)):
    print('Error...')
    exit(-1)
  for item in real_entities:
    if 'B-' in item:
      df_results.loc['total',item[2:]] += 1

  for item in range(len(real_entities)):
    if 'B-' in real_entities[item]:
      ent_start = item
      if (ent_start == len(real_entities) - 1) or ('I-' not in real_entities[item+1]) :
        ent_end = ent_start
      else:
        if item + 1 == len(real_entities) - 1:
          ent_end = item + 1
          if real_entities[ent_start:ent_end+1] == predicted_entities[ent_start:ent_end+1]:
            df_results.loc[index,real_entities[ent_start][2:]] += 1
          break
        for item_int in range(ent_start+1,len(real_entities) - 1):
          if real_entities[item_int+1] != real_entities[item_int]:
            ent_end = item_int
            break
          if item_int + 1 == len(real_entities) - 1:
            ent_end = item_int + 1
      if real_entities[ent_start:ent_end+1] == predicted_entities[ent_start:ent_end+1]:
        df_results.loc[index,real_entities[ent_start][2:]] += 1
  for item in range(len(predicted_entities)):
    if 'B-' in predicted_entities[item]:
      ent_start = item
      if (ent_start == len(predicted_entities) - 1) or ('I-' not in predicted_entities[item+1]) :
        ent_end = ent_start
      else:
        if item + 1 == len(predicted_entities) - 1:
          ent_end = item + 1
          if predicted_entities[ent_start:ent_end+1] == real_entities[ent_start:ent_end+1]:
            break
        for item_int in range(ent_start+1,len(predicted_entities) - 1):
          if predicted_entities[item_int+1] != predicted_entities[item_int]:
            ent_end = item_int
            break
          if item_int + 1 == len(predicted_entities) - 1:
            ent_end = item_int + 1
      if real_entities[ent_start:ent_end+1] != predicted_entities[ent_start:ent_end+1]:
        df_results.loc['missed',predicted_entities[ent_start][2:]] += 1
  return df_results

def find_named_entities(dictionary,N,nlp_spacy_rob,df,tool_name):
  """
  Performs entity extraction using tool and measure time elapsed.
  
  Args:
      dictionary(dict): Dictionary where results are added
      N (int): Number of texts where entity extraction is performed
      nlp_spacy_rob (): Tool for entity extraction
      df (pd.DataFrame): dataframe with texts
      tool_name (str): Tool name (e.g. 'spaCy' or 'Flair')
  Returns:
      'dictionary' updated, start and end time
  """
  start = time.time()
  for i in range(N):    #iterate through dataset
    doc = nlp_spacy_rob(df.loc[i,'sentence_non_tokenized'])
    dictionary_arr = []
    dictionary_arr.append({'sentence':str(doc)})
    for entity in doc.ents:
      for match_ in re.finditer(str(entity),str(doc)):
        dict_str = str(match_.start()) + '-' + str(match_.end())
        dictionary_arr.append({dict_str:entity.label_})
    dict_str = tool_name + '-' + str(i)
    dictionary.update({dict_str:dictionary_arr})
  end = time.time()
  return start,end,dictionary

def evaluate_tool(N, names_df, nlp_spacy_rob, df):
  """
  Evaluates tool on the set of N texts using 'add_result_to_df' for each text.
  
  Args:
      N (int): Number of texts where entity extraction is performed
      names_df (list): Entity types supported (LOC,ORG etc.)
      nlp_spacy_rob (): Tool to be evaluated
      df (pd.DataFrame): dataframe with texts
  Returns:
      Dataframe containing evaluation results
  """
  rows_indices = list(range(N)) + ['total'] + ['missed']
  df_results = pd.DataFrame(index = rows_indices, columns = names_df).fillna(0)
  for i in range(N):    #iterate through dataset
    doc = nlp_spacy_rob(df.loc[i,'sentence_non_tokenized'])
    entity_list = ['O' for item in df.loc[i,'sentences']]
    for entity in doc.ents:   #for every entity found in spacy
      index = str(doc)[:str(doc).index(str(entity))]
      index = index.count(' ') - 1
      for label_span in range(len(str(entity).split())):
        if label_span == 0:
          entity_list[index + label_span] = 'B-' + entity.label_    #construct spacy entity list as it is in dataset
        else:
          entity_list[index + label_span] = 'I-' + entity.label_
    df_results = add_result_to_df(i,df_results, entity_list, df.loc[i,'named_entities'])
  return df_results

def measure_write_result(df_results, dictionary, tool_name, start, end):
  """
  Write evaluation results from 'df_results' on dictionary
  
  Args:
      df_results (pd.DataFrame): dataframe containing evaluation results
      dictionary (dict): dictionary to be written
      tool_name (str): Name of tool to be evaluated (e.g. 'spaCy')
      start (int): Start time before entity extraction
      end(int): Ending time after entity extraction
   
  Returns:
      Dictionary updated
  """
  hit_percent_sum = 0
  hit_percent_sum_prec = 0
  total_categories = 0
  total_categories_prec = 0
  dictionary_arr = []
  for index,row in df_results.items():
    hits = row[:-2].sum()
    total = df_results.loc['total', index]
    misses = df_results.loc['missed', index]
    if total != 0 or misses != 0:
      if total > 0:
        hit_percent = hits/total  #recall
      else:
        hit_percent = 0
      if hits+misses > 0:
        hit_percent_precision = hits/(hits+misses)  #precision
      else:
        hit_percent_precision = 0
      if hit_percent > 0 and hit_percent_precision > 0:
        hit_f1 = 2/((1/hit_percent) + (1/hit_percent_precision))  #F1
      else:
        hit_f1 = 0.0

      hit_percent_sum += total*hit_percent
      hit_percent_sum_prec += (hits+misses)*hit_percent_precision

      total_categories += total
      total_categories_prec += (hits+misses)

      print('Recall: Found ', hits, ' hits out of ', total, ' for category' , index, '(', hit_percent*100, '%)')
      print('Precision: Found ', hits, ' hits out of ', hits + misses, ' for category' , index, '(', hit_percent_precision*100, '%)')
      print('F1: for category' , index, '(', hit_f1*100, '%)\n')
      str_ = 'RECALL-' + index
      str_prec = 'PRECISION-' + index
      str_f1 = 'F1-' + index
      dictionary_arr.append({str_:hit_percent*100})
      dictionary_arr.append({str_prec:hit_percent_precision*100})
      dictionary_arr.append({str_f1:hit_f1*100})
  dictionary_arr.append({'RECALL-TOTAL':hit_percent_sum*100/total_categories})
  dictionary_arr.append({'PRECISION-TOTAL':hit_percent_sum_prec*100/total_categories_prec})
  f1_total = 2/((1/(hit_percent_sum*100/total_categories)) + (1/(hit_percent_sum_prec*100/total_categories_prec)))
  dictionary_arr.append({'F1-TOTAL':f1_total})
  dictionary_arr.append({'RUNTIME':end-start})
  dict_str = 'Evaluation-' + tool_name
  dictionary.update({dict_str:dictionary_arr})
  print('Average recall hit rate: ', hit_percent_sum*100/total_categories,'%')
  print('Average precision hit rate: ', hit_percent_sum_prec*100/total_categories_prec,'%')
  print('Average F1 score: ',f1_total,'%')
  
  return dictionary

def find_named_entities_flair(dictionary,N,tagger,df):
  """
  Same as 'find_named_entities' function, but specified for Flair tool.
  
  Args:
      dictionary(dict): Dictionary where results are added
      N (int): Number of texts where entity extraction is performed
      tagger (): Tool for entity extraction
      df (pd.DataFrame): dataframe with texts
  Returns:
      'dictionary' updated, start and end time
  """
  start = time.time()
  for i in range(N):    #iterate through dataset
    sentence_tokenized = df.loc[i,'sentences']
    sentence_to_predict = df.loc[i,'sentence_non_tokenized']
    sentence = Sentence(sentence_to_predict)
    dictionary_arr = []
    dictionary_arr.append({'sentence':df.loc[i,'sentence_non_tokenized']})
    tagger.predict(sentence)
    for lb in sentence.labels:
      txt = lb.data_point.text
      tag = lb.data_point.tag
      if tag == 'PER':
        tag = 'PERSON'
      for match_ in re.finditer(str(txt),df.loc[i,'sentence_non_tokenized']):
        dict_str = str(match_.start()) + '-' + str(match_.end())
        dictionary_arr.append({dict_str:tag})
    dict_str = 'Flair-' + str(i)
    dictionary.update({dict_str:dictionary_arr})
  end = time.time()
  return start, end, dictionary

def evaluate_flair(N, names_df, tagger, df):
  """
  Same as 'evaluate_tool' function, but specified for Flair tool
  
  Args:
      N (int): Number of texts where entity extraction is performed
      names_df (list): Entity types supported (LOC,ORG etc.)
      tagger (): Flair tool to be evaluated
      df (pd.DataFrame): dataframe with texts
   
  Returns:
      Dataframe containing evaluation results
  """
  rows_indices = list(range(N)) + ['total'] + ['missed']
  df_results = pd.DataFrame(index = rows_indices, columns = names_df).fillna(0)
  for i in range(N):
    sentence_tokenized = df.loc[i,'sentences']
    sentence_to_predict = df.loc[i,'sentence_non_tokenized']
    sentence = Sentence(sentence_to_predict)
    tagger.predict(sentence)
    entity_list = ['O' for len in range(len(sentence_tokenized))]
    for lb in sentence.labels:
      txt = lb.data_point.text
      tag = lb.data_point.tag
      if tag == 'PER':
        tag = 'PERSON'
      first = 1
      index = str(sentence_to_predict)[:str(sentence_to_predict).index(str(txt))]
      index = index.count(' ') - 1
      for label_span in range(len(txt.split())):
        if label_span == 0:
          entity_list[index + label_span] = 'B-' + tag    #construct spacy entity list as it is in dataset
        else:
          entity_list[index + label_span] = 'I-' + tag
    df_results = add_result_to_df(i,df_results, entity_list, df.loc[i,'named_entities'])
  return df_results

def find_named_entities_stanza(dictionary,N,nlp_stanza,df):
  """
  Same as 'find_named_entities' function, but specified for Stanza (StanfordNLP) tool.
  
  Args:
      dictionary(dict): Dictionary where results are added
      N (int): Number of texts where entity extraction is performed
      nlp_stanza (): Tool for entity extraction
      df (pd.DataFrame): dataframe with texts
  Returns:
      'dictionary' updated, start and end time
  """
  start = time.time()
  for i in range(N):  #iterate through dataset
    doc = nlp_stanza(df.loc[i,'sentence_non_tokenized'])
    dictionary_arr = []
    dictionary_arr.append({'sentence':df.loc[i,'sentence_non_tokenized']})
    for entity in doc.entities:
      for match_ in re.finditer(str(entity.text),df.loc[i,'sentence_non_tokenized']):
        dict_str = str(match_.start()) + '-' + str(match_.end())
        dictionary_arr.append({dict_str:entity.type})
    dict_str = 'Stanza-' + str(i)
    dictionary.update({dict_str:dictionary_arr})
  end = time.time()
  return start, end, dictionary

def evaluate_stanza(N, names_df, nlp_stanza, df):
  """
  Same as 'evaluate_tool' function, but specified for Stanza (StanfordNLP) tool
  
  Args:
      N (int): Number of texts where entity extraction is performed
      names_df (list): Entity types supported (LOC,ORG etc.)
      nlp_stanza (): Stanza tool to be evaluated
      df (pd.DataFrame): dataframe with texts
   
  Returns:
      Dataframe containing evaluation results
  """
  rows_indices = list(range(N)) + ['total'] + ['missed']
  df_results = pd.DataFrame(index = rows_indices, columns = names_df).fillna(0)
  for i in range(N):    #iterate through dataset
    txt = df.loc[i,'sentence_non_tokenized']
    doc = nlp_stanza(txt)
    entity_list = ['O' for item in df.loc[i,'sentences']]
    for entity in doc.entities:
      index = str(txt)[:str(txt).index(entity.text)]
      index = index.count(' ') - 1
      for label_span in range(len(entity.text.split())):
        if label_span == 0:
          entity_list[index + label_span] = 'B-' + entity.type
        else:
          entity_list[index + label_span] = 'I-' + entity.type
    df_results = add_result_to_df(i,df_results, entity_list, df.loc[i,'named_entities'])
  return df_results

def write_json_file(dictionary,out_file_name, config_file):
  """
  Write dictionary, which contains results and optionally evaluation metrics, to a json file in directory specified in configuration file.
  
  Args:
      dictionary (dict): Dictionary which will be written on JSON file
      out_file_name (str): Path where JSON file will be created and written
      config_file (str): Configuration file
  """
  json_object = json.dumps(dictionary, indent=4)
  col_name_config,col_name_optional_config,csv_delimiter,output_path = read_config_file(config_file)
  out_file = output_path[1:-1] + '/' + out_file_name
  with open(out_file, "w") as outfile:
    outfile.write(json_object)

