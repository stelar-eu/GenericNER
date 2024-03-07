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
from minio import Minio

def df_add_value(df, idx, col, value):
  df.loc[idx,col] = value
  return df

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
  df_list = []
  for i in range(len(data_train)):
    df_temp = pd.DataFrame.from_dict(data_train[i])
    df_list.append(df_temp)
  df = pd.concat(df_list, ignore_index = True)
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

def input_data_config_file(config_file):
  """
  Reads configuration file. 
  
  Args:
      config_file (str): The configuration file to be read.
  
  Returns:
      Configuration file input data
  """
  config = ConfigParser()
  config.read(config_file)
  config_data = config['DEFAULT']
  col_name_config = config_data['namecolumn']
  col_name_optional_config = config_data['namecolumn_optional']
  csv_delimiter = config_data['csv_delimiter']
  csv_delimiter = csv_delimiter[1:-1]
  input_file = config_data['input_file']
  input_file = input_file[1:-1]
  return col_name_config,col_name_optional_config,csv_delimiter, input_file

def get_output_config_file(config_file):
  config = ConfigParser()
  config.read(config_file)
  config_data = config['DEFAULT']
  output_file_name = config_data['output_file_name'][1:-1]
  output_path = config_data['output_file_path']
  return output_file_name,output_path

def check_input_with_config(input_file_path, text_column, ground_truth_column, csv_delimiter, output_file_path, minio):
  """
  Checks if input CSV file has the correct format, based on configuration file.
  
  Args:
      conf_file (str): The configuration file to be read.
      list_arguments (list): Arguments passed when running the progam.
  Returns:
      Configuration file data and dataframe containing CSV file data
  """
  col_name_config = text_column
  col_name_optional_config = ground_truth_column
  output_path  = output_file_path	#check if given csv in correct form
  df_input = pd.DataFrame()
  if '.csv' not in input_file_path:
    df_input == pd.DataFrame()
    
  if input_file_path.startswith('s3://'):
    bucket, key = input_file_path.replace('s3://', '').split('/', 1)
    client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
    df_input = pd.read_csv(client.get_object(bucket, key), delimiter = csv_delimiter)
  else:
    df_input = pd.read_csv(input_file_path, delimiter = csv_delimiter)
    
  return df_input,col_name_config,col_name_optional_config

def prepare_given_dataset(df_input, col_name_config, col_name_optional_config):
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
    if len(df_input.columns) == 2 and df_input.columns[1] == col_name_optional_config:
      df['named_entities'] = df_input[col_name_optional_config]
      df['named_entities'] = df['named_entities'].apply(lambda x: x[2:-2].split('\',\''))
      for i in range(len(df)):
        if len(df['named_entities'][i]) != len(df['sentence_non_tokenized'][i].split()):
          print('Error: the number of tags must be the same as the number of tokens in every text.')
          df = pd.DataFrame()
  return df

def make_df_by_argument(input_file_path, text_column, ground_truth_column, namecolumn_optional_2 , csv_delimiter, minio):
  """
  Returns dataframe according to whether a CSV file was given as an input or not.
  
  Args:
      list_arguments (list): List of arguments
      config_file (str): Configuration file
  
  Returns:
      The dataframe
  """
  df_input,col_name_config,col_name_optional_config  = check_input_with_config(input_file_path, text_column, ground_truth_column, csv_delimiter, output_file_path, minio)
  if df_input.empty:
    print('ERROR: Please provide a .csv file.')
    exit()  
  df = prepare_given_dataset(df_input, col_name_config,col_name_optional_config,namecolumn_optional_2) 
  if df.empty:
    print('ERROR: Wrong CSV format!')
    exit()
  return df

def choose_num(df, default):
  """
  How many texts will have their entities labelled. Defaults to 'default'. If 'default' is bigger than the number of texts, the whole dataset will be used for entity extraction.
  
  Args:
      df (pd.DataFrame): Dataframe containing texts
      default (int or str): Number of texts for entity extraction or 'all' for the whole dataset
  
  Returns:
      Number chosen
  """
  if (type(default) == str and default == 'all') or (type(default) == int and default > len(df)):
    N = len(df)
  else:
    N = default
  return N

def load_import_models():
  """
  Load entity extraction models
  
  Returns:
      Models loaded
  """  
  spacy.prefer_gpu()
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
  return [nlp_spacy_rob,nlp_spacy,tagger,nlp_stanza]

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
      df_results.loc['positive',item[2:]] += 1

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
        if real_entities[ent_start:ent_end+1] == ['O'for it in range(len(real_entities[ent_start:ent_end+1]))]:
          df_results.loc['fp',predicted_entities[ent_start][2:]] += 1
        else:
          df_results.loc['wrong_category',predicted_entities[ent_start][2:]] += 1
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
      for match_ in re.finditer(re.escape(str(entity)),str(doc)):
        dict_str = str(match_.start()) + '-' + str(match_.end())
        dictionary_arr.append({dict_str:entity.label_})
    dict_str = tool_name + '-' + str(i)
    dictionary.update({dict_str:dictionary_arr})
  end = time.time()
  print('Finished entity recognition by ', tool_name, 'in ', end-start)
  return start,end,dictionary

def return_bio_format(N, nlp_spacy_rob, df, tool_name):
   '''
    Add description
   '''
   results_list = []
   name = 'spaCy'
   if 'RoBERTa' in tool_name:
    name += ' + RoBERTa'
   for i in range(N):    #iterate through dataset
    doc = nlp_spacy_rob(df.loc[i,'sentence_non_tokenized'])
    entity_list = ['O' for item in df.loc[i,'sentences']]
    doc_len = len(str(doc))
    new_doc = doc
    for entity in doc.ents:   #for every entity found in spacy
      index_entity = doc_len - len(str(new_doc)) + str(new_doc).index(str(entity))
      index = str(doc)[:index_entity].count(' ') - 1
      if str(doc)[:index_entity][-1] == ' ':
        index = len(str(doc)[:index_entity].split())
      else:
        index = len(str(doc)[:index_entity].split()) - 1
      new_doc = str(doc)[index_entity + len(str(entity)):]
      for label_span in range(len(str(entity).split())):
        if label_span == 0:
          entity_list[index + label_span] = 'B-' + entity.label_    #construct spacy entity list as it is in dataset
        else:
          entity_list[index + label_span] = 'I-' + entity.label_
    results_list.append(entity_list)
   df_results_total = pd.DataFrame({name:results_list})
   return df_results_total

def compare_results(total_results_df,N,names_df):
  print(N)

def evaluate_tool(N, names_df, nlp_spacy_rob, df, tool_name):
  """
  Evaluates tool on the set of N texts using 'add_result_to_df' for each text.
  
  Args:
      N (int): Number of texts where entity extraction is performed
      names_df (list): Entity types supported (LOC,ORG etc.)
      nlp_spacy_rob (): Tool to be evaluated
      df (pd.DataFrame): dataframe with texts
      tool_name (str): name of tool (e.g. 'spaCy' or 'spaCy + RoBERTa')
  Returns:
      Dataframe containing evaluation results
  """
  results_list = []
  rows_indices = list(range(N)) + ['positive'] + ['fp'] + ['wrong_category']
  df_results = pd.DataFrame(index = rows_indices, columns = names_df).fillna(0)
  name = 'spaCy'
  if 'RoBERTa' in tool_name:
    name += ' + RoBERTa'
  for i in range(N):    #iterate through dataset
    doc = nlp_spacy_rob(df.loc[i,'sentence_non_tokenized'])
    entity_list = ['O' for item in df.loc[i,'sentences']]
    doc_len = len(str(doc))
    new_doc = doc
    for entity in doc.ents:   #for every entity found in spacy
      index_entity = doc_len - len(str(new_doc)) + str(new_doc).index(str(entity))
      index = str(doc)[:index_entity].count(' ') - 1
      if str(doc)[:index_entity][-1] == ' ':
        index = len(str(doc)[:index_entity].split())
      else:
        index = len(str(doc)[:index_entity].split()) - 1
      new_doc = str(doc)[index_entity + len(str(entity)):]
      for label_span in range(len(str(entity).split())):
        if label_span == 0:
          entity_list[index + label_span] = 'B-' + entity.label_    #construct spacy entity list as it is in dataset
        else:
          entity_list[index + label_span] = 'I-' + entity.label_
    results_list.append(entity_list)
    df_results = add_result_to_df(i,df_results, entity_list, df.loc[i,'named_entities'])
  df_results_total = pd.DataFrame({name:results_list})
  return df_results, df_results_total

def merge_dfs_horizontally(dfs_list):
  final_df = pd.concat(dfs_list, axis=1)
  return final_df

def cross_results(df,N,names_df, dictionary):
  '''
  add description
  '''
  print('------------------------------------------------------------')
  print('Models\' agreement')
  df_final_f1 = pd.DataFrame(index = list(df.columns), columns = list(df.columns))
  df_final_prec = pd.DataFrame(index = list(df.columns), columns = list(df.columns))
  df_final_rec = pd.DataFrame(index = list(df.columns), columns = list(df.columns))
  df_wrong_category = pd.DataFrame(index = list(df.columns), columns = list(df.columns))
  rows_indices = list(range(N)) + ['positive'] + ['fp']  + ['wrong_category']
  dictionary_arr = []
  for tool in list(df.columns):
    for tool_2 in list(df.columns):
      df_results = pd.DataFrame(index = rows_indices, columns = names_df).fillna(0)
      for i in range(N):
        df_results = add_result_to_df(i, df_results, df.loc[i,tool], df.loc[i,tool_2])
      print('df_results:',df_results)
      hit_percent_sum = 0
      hit_percent_sum_prec = 0
      total_categories = 0
      total_categories_prec = 0
      perc_wrong_sum = 0
      for index,row in df_results.items():
        hits = row[:-3].sum()
        total = df_results.loc['positive', index]
        misses = df_results.loc['fp', index]
        wrong_category = df_results.loc['wrong_category', index]
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
          perc_wrong = wrong_category/total  #wrongly labelled

          hit_percent_sum += total*hit_percent
          hit_percent_sum_prec += (hits+misses)*hit_percent_precision

          total_categories += total
          total_categories_prec += (hits+misses)
          perc_wrong_sum += perc_wrong
      try:
        f1_total = 2/((1/(hit_percent_sum*100/total_categories)) + (1/(hit_percent_sum_prec*100/total_categories_prec)))
      except:
        f1_total = 0
      df_final_f1.loc[tool,tool_2] = str(f1_total) + '%'
      prec_total = hit_percent_sum_prec*100/total_categories_prec
      df_final_prec.loc[tool,tool_2] = str(prec_total) + '%'
      rec_total = hit_percent_sum*100/total_categories
      df_final_rec.loc[tool,tool_2] = str(rec_total) + '%'
      perc_wrong_total = perc_wrong_sum*100/total_categories
      df_wrong_category.loc[tool,tool_2] = str(perc_wrong_total) + '%'
      str_ = tool +'-' + tool_2 + '-'
      dictionary.update({'Cross results-' + str_ + 'F1':df_final_f1.loc[tool,tool_2]})
      dictionary.update({'Cross results-' + str_ + 'precision':df_final_prec.loc[tool,tool_2]})
      dictionary.update({'Cross results-' + str_ + 'recall':df_final_rec.loc[tool,tool_2]})
      dictionary.update({'Cross results-' + str_ + 'wrong category':df_wrong_category.loc[tool,tool_2]})
  print('F1',df_final_f1)
  print('Precision', df_final_prec)
  print('Recall', df_final_rec)
  print('wrong category', df_wrong_category)
  return dictionary

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
    hits = row[:-3].sum()
    total = df_results.loc['positive', index]
    misses = df_results.loc['fp', index]
    if total != 0 or hits + misses != 0:
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
      str_ = tool_name + '-RECALL-' + index
      str_prec = tool_name +'-PRECISION-' + index
      str_f1 = tool_name +'-F1-' + index
      dictionary.update({str_:hit_percent*100})
      dictionary.update({str_prec:hit_percent_precision*100})
      dictionary.update({str_f1:hit_f1*100})
  dictionary.update({tool_name +'-RECALL-TOTAL':hit_percent_sum*100/total_categories})
  dictionary.update({tool_name +'-PRECISION-TOTAL':hit_percent_sum_prec*100/total_categories_prec})
  f1_total = 2/((1/(hit_percent_sum*100/total_categories)) + (1/(hit_percent_sum_prec*100/total_categories_prec)))
  dictionary.update({tool_name +'-F1-TOTAL':f1_total})
  dictionary.update({tool_name +'-RUNTIME':end-start})
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
      for match_ in re.finditer(re.escape(str(txt)),df.loc[i,'sentence_non_tokenized']):
        dict_str = str(match_.start()) + '-' + str(match_.end())
        dictionary_arr.append({dict_str:tag})
    dict_str = 'Flair-' + str(i)
    dictionary.update({dict_str:dictionary_arr})
  end = time.time()
  print('Finished entity recognition by Flair in ', end-start)
  return start, end, dictionary

def return_bio_format_flair(N, tagger, df):
   results_list = []
   for i in range(N):    #iterate through dataset
    sentence_tokenized = df.loc[i,'sentences']
    sentence_to_predict = df.loc[i,'sentence_non_tokenized']
    sentence = Sentence(sentence_to_predict)
    tagger.predict(sentence)
    entity_list = ['O' for len in range(len(sentence_tokenized))]
    doc_len = len(sentence_to_predict)
    new_doc = sentence_to_predict
    for lb in sentence.labels:
      txt = lb.data_point.text
      tag = lb.data_point.tag
      if tag == 'PER':
        tag = 'PERSON'
      first = 1
      try:
        index_entity = doc_len - len(str(new_doc)) + str(new_doc).index(str(txt))
      except:
        print('exception')
        continue
      index = sentence_to_predict[:index_entity].count(' ') - 1
      if sentence_to_predict[:index_entity][-1] == ' ':
        index = len(sentence_to_predict[:index_entity].split())
      else:
        index = len(sentence_to_predict[:index_entity].split()) - 1
      new_doc = sentence_to_predict[index_entity + len(str(txt)):]
      for label_span in range(len(txt.split())):
        if label_span == 0:
          entity_list[index + label_span] = 'B-' + tag    #construct spacy entity list as it is in dataset
        else:
          entity_list[index + label_span] = 'I-' + tag
    results_list.append(entity_list)
   df_results_total = pd.DataFrame({'Flair':results_list})
   return df_results_total

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
  rows_indices = list(range(N)) + ['positive'] + ['fp']  + ['wrong_category']
  df_results = pd.DataFrame(index = rows_indices, columns = names_df).fillna(0)
  results_list = []
  for i in range(N):
    sentence_tokenized = df.loc[i,'sentences']
    sentence_to_predict = df.loc[i,'sentence_non_tokenized']
    sentence = Sentence(sentence_to_predict)
    tagger.predict(sentence)
    entity_list = ['O' for len in range(len(sentence_tokenized))]
    doc_len = len(str(doc))
    new_doc = doc
    for lb in sentence.labels:
      txt = lb.data_point.text
      tag = lb.data_point.tag
      if tag == 'PER':
        tag = 'PERSON'
      first = 1
      index_entity = doc_len - len(str(new_doc)) + str(new_doc).index(str(txt))
      index = str(sentence)[:index_entity].count(' ') - 1
      if sentence_to_predict[:index_entity][-1] == ' ':
        index = len(sentence_to_predict[:index_entity].split())
      else:
        index = len(sentence_to_predict[:index_entity].split()) - 1
      new_doc = str(sentence)[index_entity + len(txt):]
      for label_span in range(len(txt.split())):
        if label_span == 0:
          entity_list[index + label_span] = 'B-' + tag    #construct spacy entity list as it is in dataset
        else:
          entity_list[index + label_span] = 'I-' + tag
    results_list.append(entity_list)
    df_results = add_result_to_df(i,df_results, entity_list, df.loc[i,'named_entities'])
  df_results_total = pd.DataFrame({'Flair':results_list})
  return df_results, df_results_total

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
      for match_ in re.finditer(re.escape(str(entity.text)),df.loc[i,'sentence_non_tokenized']):
        dict_str = str(match_.start()) + '-' + str(match_.end())
        dictionary_arr.append({dict_str:entity.type})
    dict_str = 'Stanza-' + str(i)
    dictionary.update({dict_str:dictionary_arr})
  end = time.time()
  print('Finished entity recognition by Stanza in ', end-start)
  return start, end, dictionary

def return_bio_format_stanza(N, nlp_stanza, df):
   results_list = []
   for i in range(N):    #iterate through dataset
    txt = df.loc[i,'sentence_non_tokenized']
    doc = nlp_stanza(txt)
    entity_list = ['O' for item in df.loc[i,'sentences']]
    new_doc = txt
    doc_len = len(txt)
    for entity in doc.entities:
      index_entity = doc_len - len(str(new_doc)) + str(new_doc).index(entity.text)
      index = str(txt)[:index_entity].count(' ') - 1
      if txt[:index_entity][-1] == ' ':
        index = len(txt[:index_entity].split())
      else:
        index = len(txt[:index_entity].split()) - 1
      new_doc = str(txt)[index_entity + len(str(entity.text)):]
      for label_span in range(len(entity.text.split())):
        if label_span == 0:
          entity_list[index + label_span] = 'B-' + entity.type
        else:
          entity_list[index + label_span] = 'I-' + entity.type
    results_list.append(entity_list)
   df_results_total = pd.DataFrame({'Stanza':results_list})
   return df_results_total

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
  rows_indices = list(range(N)) + ['positive'] + ['fp'] + ['wrong_category']
  df_results = pd.DataFrame(index = rows_indices, columns = names_df).fillna(0)
  results_list = []
  for i in range(N):    #iterate through dataset
    txt = df.loc[i,'sentence_non_tokenized']
    doc = nlp_stanza(txt)
    entity_list = ['O' for item in df.loc[i,'sentences']]
    doc_len = len(str(doc))
    new_doc = doc
    for entity in doc.entities:
      index_entity = doc_len - len(str(new_doc)) + str(new_doc).index(entity.text)
      index = str(txt)[:index_entity].count(' ') - 1
      if txt[:index_entity][-1] == ' ':
        index = len(txt[:index_entity].split())
      else:
        index = len(txt[:index_entity].split()) - 1
      new_doc = str(txt)[index_entity + len(str(entity.text)):]
      for label_span in range(len(entity.text.split())):
        if label_span == 0:
          entity_list[index + label_span] = 'B-' + entity.type
        else:
          entity_list[index + label_span] = 'I-' + entity.type
    results_list.append(entity_list)
    df_results = add_result_to_df(i,df_results, entity_list, df.loc[i,'named_entities'])
  df_results_total = pd.DataFrame({'Stanza':results_list})
  return df_results, df_results_total

def write_json_file(dictionary,out_file_name):
  """
  Write dictionary, which contains results and optionally evaluation metrics, to a json file in directory specified in configuration file.
  
  Args:
      dictionary (dict): Dictionary which will be written on JSON file
      out_file_name (str): Path where JSON file will be created and written
      config_file (str): Configuration file
  """
  json_object = json.dumps(dictionary, indent=4)
  with open(out_file_name + '.json', "w") as outfile:
    outfile.write(json_object)

