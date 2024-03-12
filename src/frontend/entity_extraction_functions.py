from annotated_text import annotated_text
import streamlit as st
import pandas as pd
import numpy as np
import wikipedia
import re
import json
import time

def make_json_object():
  json_file = st.sidebar.file_uploader('Upload json file', type = 'json')
  if json_file is not None:
    # Reading from json file
    json_object = json.load(json_file)
  else:
    while True:
      time.sleep(5)   # Delay for 5 seconds
  return json_object

def make_tools_list(json_object):
  tools_list = []
  for key_json,value_json in json_object.items():
    if key_json != 'Cross results' and key_json[:key_json.index('-')] not in tools_list and 'Evaluation' not in key_json:
      tools_list.append(key_json[:key_json.index('-')])
  return tools_list

def select_index(form, json_object, n_tools):
  form.num_select = st.selectbox(
      'By index', 
      range(1,int((len(json_object)-n_tools)/n_tools)+1))
  num_select = form.num_select
  num = form.num_select - 1
  return num, num_select

def select_options_type_filter(form, list_categories):
  options_type_filter = []
  form.options_type_filter = st.multiselect(
    'By filtering',
    list_categories)  
  options_type_filter = form.options_type_filter
  return options_type_filter

def select_keyword(form):
  form.kw = st.text_input('By keyword')
  kw = form.kw
  return kw

def choose_entity_type(list_categories):
  options_type = []
  st.sidebar.write('Choose entity type to extract')
  for tool in ['ALL'] + list_categories:
    checkbox = st.sidebar.checkbox(tool)
    if checkbox:
      options_type.append(tool)
  return options_type

def choose_tools(tools_list):
  st.sidebar.write('Choose preferred tool for entity extraction')
  tools = []
  for tool in tools_list: 
    if tool == 'All':
      checkbox = st.sidebar.checkbox(tool, value = True)
    else:
      checkbox = st.sidebar.checkbox(tool)
    if checkbox:
      tools.append(tool)
  return tools

def annotate_text(sentence, keys, values, all_or_not, options_type = ['ALL','ALL']):
  list_tuples = [] 
  annotated = [0 for i in range(len(sentence.split()))]
  index_in_str = 1
  #index_in_str = 0		#this is right when not using OntoNotes...
  index_token = -1
  for word in sentence.split():
    index_token += 1
    is_entity = 0
    count = 0
    word_start = index_in_str
    word_end = word_start + len(word)
    dont_include = 0
    for entity_span in keys:
      span_start = int(str(entity_span)[:(str(entity_span).index('-'))])
      span_end = int(str(entity_span)[(str(entity_span).index('-')+1):])
      if word_start >= span_start and word_end <= span_end and (values[count] in options_type or 'ALL' in options_type or all_or_not == 'ALL'):
        if annotated[index_token] == 0:
          tup = (sentence[span_start:span_end], values[count])
          list_tuples.append(tup)
          for i in range(index_token,index_token + len(sentence[span_start:span_end].split())):
            try:
              annotated[i] = 1
            except:
              print(i)
        is_entity = 1
      count += 1
    if is_entity == 0:
        tup = (word,)
        list_tuples.append(tup)
    index_in_str += len(word) + 1

  if list_tuples == []:
    annotated_text(sentence)
  else:
    annotated_text(
    list_tuples
  )

def majority_voting(json_object, num):
    json_object_index = str(num)
    keys = []
    values = [] 
    candidate_entities_votes = pd.DataFrame(columns = ['votes'])
    for key_json,value_json in json_object.items():
      if key_json != 'Cross results' and json_object_index == key_json[key_json.index('-')+1:]:
        for item in json_object[key_json]:
          for key,value in item.items():
            if key != 'sentence' and key in candidate_entities_votes.index:
              candidate_entities_votes.loc[key] += 1
            elif key != 'sentence' and key not in candidate_entities_votes.index:
              candidate_entities_votes.loc[key] = 1     
    for index,row in candidate_entities_votes.iterrows():
      for index_2,row_2 in candidate_entities_votes.iterrows():
        if index != index_2:
          if (int(index[:index.index('-')]) <= int(index_2[:index_2.index('-')])) and (int(index[index.index('-')+1:]) >= int(index_2[index_2.index('-')+1:])):
            if row['votes'] >= row_2['votes']:
              candidate_entities_votes = candidate_entities_votes.drop(labels=index_2, axis=0)
            elif row['votes'] < row_2['votes']:
              candidate_entities_votes = candidate_entities_votes.drop(labels=index, axis=0)
              break
    for index,row in candidate_entities_votes.iterrows():
      for index_2,row_2 in candidate_entities_votes.iterrows():
        if index != index_2:
          if (int(index[index.index('-')+1:]) > int(index_2[:index_2.index('-')])) and (int(index[index.index('-')+1:]) < int(index_2[index_2.index('-')+1:])):
            if row['votes'] >= row_2['votes']:
              candidate_entities_votes = candidate_entities_votes.drop(labels=index, axis=0)
              continue
            elif row['votes'] < row_2['votes']:
              candidate_entities_votes = candidate_entities_votes.drop(labels=index_2, axis=0)
              break
    for key_json,value_json in json_object.items():
      if key_json != 'Cross results' and json_object_index == key_json[key_json.index('-')+1:]:
        for item in json_object[key_json]:
          for key,value in item.items():
            if key != 'sentence' and key in candidate_entities_votes.index:
              keys.append(key)
              values.append(value)
    return keys, values
   
def make_keys_values(tools, json_object, num):
  if tools is not None and len(tools) == 1 and 'All' not in tools:
    json_object_index = tools[0] + '-' +str(num)
  keys = []
  values = [] 
  if 'All' not in tools and len(tools) == 1 and tools is not None:
    for item in json_object[json_object_index]:
      for key,value in item.items():
        if key != 'sentence':
          keys.append(key)
          values.append(value)
  elif 'All' in tools:
    keys, values = majority_voting(json_object, num)
  elif len(tools) > 1 and 'All' not in tools:
    json_object_index = str(num)
    for key_json,value_json in json_object.items():
      if json_object_index == key_json[key_json.index('-')+1:] and key_json[:key_json.index('-')] in tools:
        for item in json_object[key_json]:
          for key,value in item.items():
            if key != 'sentence':
              keys.append(key)
              values.append(value)
  return keys,values

def filter_sentences(preferred,json_object, tools, options_type, tools_list): 
  found = 0
  how_many = 0
  if preferred is not None and preferred != '': 
    for key_json,value_json in json_object.items():
      if ('Cross results' != key_json) and ('Evaluation' not in key_json) and preferred is not None and ((preferred in value_json[0]['sentence']) or (preferred.lower() in value_json[0]['sentence']) or (preferred in value_json[0]['sentence'].lower())) and tools_list[1] in key_json:
        found = 1
        how_many += 1
        keys,values = make_keys_values(tools, json_object, key_json[key_json.index('-')+1:])
        st.write('Text no.',str(int(key_json[key_json.index('-')+1:])+1))
        annotate_text(value_json[0]['sentence'], keys, values, 'NOT', options_type)
    if found == 0:
      st.write('No entry found!')
    else:
      st.write('__________________\n', str(how_many), ' results')

def filter_sentences_category(preferred_names_list,preferred_types_list,json_object, tools, options_type, options_type_filter,d_start,d_end): 
  found = 0
  signal_no_entry = 0
  labels_yes = 0
  i = -1
  how_many = 0
  sentence_printed = pd.DataFrame(columns = ['is_displayed'])
  if d_start != '' or d_end != '':
    preferred_types_list.append('DATE')
  if preferred_names_list != []: 
    for preferred in preferred_names_list:
      i += 1
      for key_json,value_json in json_object.items():
        if (preferred_types_list[i] != 'DATE') and ('Cross results' != key_json) and ('Evaluation' not in key_json) and ((preferred in value_json[0]['sentence']) or (preferred.lower() in value_json[0]['sentence']) or (preferred in value_json[0]['sentence'].lower())) and 'Stanza' in key_json: 
          found = 0
          for item in json_object[key_json]:
            for key,value in item.items():
              if key != 'sentence' and value == preferred_types_list[i] and ((value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] == preferred) or preferred == '' or preferred == []):
                found = 1 
                signal_no_entry = 1 
                labels_yes = 0
                if value in options_type or 'ALL' in options_type:
                  labels_yes = 1
          if found == 1 and labels_yes == 1 and str(int(key_json[key_json.index('-')+1:])+1) not in sentence_printed.index.values:
            sentence_printed.loc[str(int(key_json[key_json.index('-')+1:])+1),'is_displayed'] = 1
            how_many += 1
            keys,values = make_keys_values(tools, json_object, key_json[key_json.index('-')+1:])
            st.write('Text no.',str(int(key_json[key_json.index('-')+1:])+1))
            try:
              annotate_text(value_json[0]['sentence'], keys, values, 'NOT', options_type)
            except:
              keys,values = make_keys_values(tools, json_object, key_json[key_json.index('-')+1:])
              annotate_text(value_json[0]['sentence'], keys, values, 'NOT', options_type)
          elif found == 1 and labels_yes == 0 and str(int(key_json[key_json.index('-')+1:])+1) not in sentence_printed.index.values:
            sentence_printed.loc[str(int(key_json[key_json.index('-')+1:])+1),'is_displayed'] = 1
            how_many += 1
            st.write('Text no.',str(int(key_json[key_json.index('-')+1:])+1))
            keys,values = make_keys_values(tools, json_object, key_json[key_json.index('-')+1:])
            annotate_text(value_json[0]['sentence'], keys, values, 'NOT', options_type)
  
  for key_json,value_json in json_object.items():
    found_date = 0
    if('DATE' in preferred_types_list) and ('Cross results' != key_json) and ('Evaluation' not in key_json) and 'Stanza' in key_json and d_start is not None and (d_start != '' or d_end != '') and str(int(key_json[key_json.index('-')+1:])+1) not in sentence_printed.index.values:
      for item in json_object[key_json]:
        for key,value in item.items():
          if key != 'sentence' and value == 'DATE':
            for word in value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])].split():
              if (word.isdigit() and d_end != '' and d_start!= '' and int(word) >= int(d_start) and int(word) <= int(d_end) and float(word) == int(word)) or (d_end == '' and word.isdigit() and int(word) >= int(d_start) and float(word) == int(word)) or (d_start == '' and word.isdigit() and int(word) <= int(d_end) and float(word) == int(word)):
                sentence_printed.loc[str(int(key_json[key_json.index('-')+1:])+1),'is_displayed'] = 1
                how_many += 1
                found_date = 1
                how_many += 1
                signal_no_entry = 1 
                keys,values = make_keys_values(tools, json_object, key_json[key_json.index('-')+1:])
                st.write('Text no.', key_json[key_json.index('-')+1:])
                annotate_text(value_json[0]['sentence'], keys, values, 'NOT', options_type)
                break
          if found_date == 1:
            break
        if found_date == 1:
          break
  if signal_no_entry == 0:
    st.write('No entry found!')
  else:
    st.write('__________________\n', str(how_many), ' results')

def print_statistics(json_object, list_categories, tools_list):
  list_num = [0 for i in range(len(list_categories))]
  dictionary = {}
  for tool in tools_list:
    dictionary['list_%s' % tool] = [0 for i in range(len(list_categories))]
  list_entities = []
  list_entities_total = []
  for key_json_old,value_json in json_object.items():
    break
  for key_json,value_json in json_object.items():
    if key_json != 'Cross results' and key_json_old[:key_json_old.index('-')] != key_json[:key_json.index('-')]:
      list_entities = []
    key_json_old = key_json
    for item in json_object[key_json]:
      for key,value in item.items():
        if key != 'sentence' and ('Evaluation' not in key_json) and key_json != 'Cross results':
          if (value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] + '--' + value) not in list_entities:
            list_entities.append(value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] + '--' + value)
            if (value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] + '--' + value) not in list_entities_total:
              list_entities_total.append(value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] + '--' + value)
              list_num[list_categories.index(value)] += 1
  for key_json_old,value_json in json_object.items():
    break
  entity_counted = [0 for i in list_entities_total]  
  for key_json,value_json in json_object.items(): 
    if key_json != 'Cross results' and key_json_old[:key_json_old.index('-')] != key_json[:key_json.index('-')]:
      entity_counted = [0 for i in list_entities_total]
    key_json_old = key_json
    for item in json_object[key_json]:
      for key,value in item.items():
        if key != 'sentence' and ('Evaluation' not in key_json) and key_json != 'Cross results':
          if (value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] + '--' + value) in list_entities_total:
            if entity_counted[list_entities_total.index(value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] + '--' + value)] == 0:     
              dictionary['list_%s' % key_json[:key_json.index('-')]][list_categories.index(value)] += 1
              entity_counted[list_entities_total.index(value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] + '--' + value)] = 1
  idx = 0
  df_data = pd.DataFrame(dictionary, index = list_categories)
  df_data = df_data.drop(columns = 'list_All')
  for tool in tools_list:
    if tool != 'All':
      df_data.rename(columns={'list_' + tool: 'Unique entities identified by' + tool},
          inplace=True, errors='raise')
  df_data['Total unique entities detected'] = list_num
  st.table(df_data)

def print_statistics_specific(json_object, num):
  found = 0
  entities = []
  for key_json,value_json in json_object.items():
    if  key_json != 'Cross results' and key_json[key_json.index('-')+1:] == str(num):
      for item in json_object[key_json]:
        for key,value in item.items():
          if key != 'sentence'and ('Evaluation' not in key_json) and value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] not in entities:
            entities.append(value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])])
   
  with st.expander('Details by entity'):
    df_ent = pd.DataFrame(index = entities)
    for entity in entities:
      #st.write(f"**{entity}**") #uncomment_1
      for key_json,value_json in json_object.items():
        if  key_json != 'Cross results' and key_json[key_json.index('-')+1:] == str(num):
          for item in json_object[key_json]:
            for key,value in item.items():
              if key != 'sentence' and value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] == entity:
                #st.write('Identified as ', value, 'by ', key_json[:key_json.index('-')]) #uncomment_2
                if key_json[:key_json.index('-')] in df_ent.columns:
                  df_ent[key_json[:key_json.index('-')]][entity] = value
                else:
                  df_ent[key_json[:key_json.index('-')]] = ''
                  df_ent[key_json[:key_json.index('-')]][entity] = value
    st.table(df_ent)		#if empty?


def print_all_category(options_type_filter, json_object, options_type, tools, tools_list):
   how_many = 0
   if tools_list != []:
     found = [0 for i in range(int(len(json_object)/(len(tools_list)-1)))]
   if tools == []:
     tools = ['All']
   for key_json_old,value_json in json_object.items():
     break
   i = -1
   for key_json,value_json in json_object.items():
     i += 1
     if key_json != 'Cross results' and key_json_old[:key_json_old.index('-')] != key_json[:key_json.index('-')]:
      i = 0
     key_json_old = key_json
     for item in json_object[key_json]:
        for key,value in item.items():
          if value in options_type_filter and ('Evaluation' not in key_json):
            if found[i] == 0:
              found[i] = 1
              how_many += 1
              keys,values = make_keys_values(tools, json_object, key_json[key_json.index('-')+1:])
              st.write('Text no.',str(int(key_json[key_json.index('-')+1:])+1))
              annotate_text(value_json[0]['sentence'], keys, values, 'NOT', options_type)
   count = 0
   for i in found: 
     count += 1
     if i == 1:
       break
   if count == len(found):
     st.write('No entry found!')
   else:
     st.write('__________________\n', str(how_many), ' results')

def give_info_entity(num,json_object):
  entities = []
  for key_json,value_json in json_object.items():
    if  key_json != 'Cross results' and key_json[key_json.index('-')+1:] == str(num):
      for item in json_object[key_json]:
        for key,value in item.items():
          if key != 'sentence' and value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])] not in entities:
            entities.append(value_json[0]['sentence'][int(key[:key.index('-')]):int(key[key.index('-')+1:])])
  for ent in entities:
    entity_select = st.checkbox(
       ent)
    if entity_select:
      st.write(wikipedia.page(ent).url)
      st.write(wikipedia.summary(ent))

def print_evaluation(json_object, list_categories, tools_list):
  found = 0
  idxs = list_categories + ['TOTAL']
  df_precision = pd.DataFrame(index = idxs, columns = tools_list[1:])
  df_recall = pd.DataFrame(index = idxs, columns = tools_list[1:])
  df_f1 = pd.DataFrame(index = idxs, columns = tools_list[1:])		#! 1:
  df_runtimes = pd.DataFrame(index = tools_list[1:], columns = ['RUNTIME'])
  for key_json, value_json in json_object.items():
    if 'Evaluation' in key_json:
      found = 1
      for item in json_object[key_json]:
        for key, value in item.items():
          if 'PRECISION' in key:
            df_precision[key_json[key_json.index('-')+1:]][key[key.index('-')+1:]] = "{:.2f}".format(value) + '%'
          elif 'RECALL' in key:
            df_recall[key_json[key_json.index('-')+1:]][key[key.index('-')+1:]] = "{:.2f}".format(value) + '%'
          elif 'F1' in key:
            df_f1[key_json[key_json.index('-')+1:]][key[key.index('-')+1:]] = "{:.2f}".format(value) + '%'
          elif 'RUNTIME' in key:
            if value >= 60:
              df_runtimes['RUNTIME'][key_json[key_json.index('-')+1:]] = "{:.2f}".format(value/60) + ' minutes'
            else:
              df_runtimes['RUNTIME'][key_json[key_json.index('-')+1:]] = "{:.2f}".format(value) + ' seconds'
  df_precision = df_precision.fillna('-')
  df_recall = df_recall.fillna('-')
  df_f1 = df_f1.fillna('-')
  if found == 1:
    st.write('Precision score')
    st.table(df_precision)
    st.write('Recall score')
    st.table(df_recall)
    st.write('F1 score')
    st.table(df_f1)
    st.write('Time elapsed for labelling entities of ', str(int((len(json_object)-len(tools_list)+1)/(len(tools_list)-1))), ' texts')
    st.table(df_runtimes)
  else:
    st.write('No ground truth provided for evaluation!')

def print_cross(json_object, list_categories, tools_list):
  df_f1 = pd.DataFrame(index = tools_list[1:], columns = tools_list[1:])	#! hard coded
  df_prec = pd.DataFrame(index = tools_list[1:], columns = tools_list[1:])
  df_rec = pd.DataFrame(index = tools_list[1:], columns = tools_list[1:])
  df_wrong_category = pd.DataFrame(index = tools_list[1:], columns = tools_list[1:])
  for key_json, value_json in json_object.items():
    if key_json == 'Cross results':
      for item in json_object[key_json]:
        for key, value in item.items():
          if 'F1' in key:
            df_f1.loc[key[:key.index('-')],key[key.index('-')+1:key.index('F1')-1]] = value
          elif 'precision' in key:
            df_prec.loc[key[:key.index('-')],key[key.index('-')+1:key.index('precision')-1]] = value
          elif 'recall' in key:
            df_rec.loc[key[:key.index('-')],key[key.index('-')+1:key.index('recall')-1]] = value
          elif 'wrong category' in key:
            df_wrong_category.loc[key[:key.index('-')],key[key.index('-')+1:key.index('wrong category')-1]] = value
  st.write('F1 score', df_f1)
  st.write('Recall score', df_rec)
  st.write('Precision score', df_prec)
  st.write('Different categories predicted', df_wrong_category)

def preferred_entity(options_type_filter,form):
  preferred_name = []
  preferred_type = [] 
  found_type = 0
  if 'GPE' in options_type_filter:
    found_type = 1
    form.vari = st.text_input('Choose country or city')
    vari = form.vari
    preferred_name.append(vari)
    preferred_type.append('GPE')
    options_type_filter.remove('GPE')
  if 'LOC' in options_type_filter:
    found_type = 1
    form.vari = st.text_input('Choose location (mountain, river etc.)')
    vari = form.vari
    preferred_name.append(vari)
    preferred_type.append('LOC')
    options_type_filter.remove('LOC')
  if 'EVENT' in options_type_filter:
    found_type = 1
    form.vari = st.text_input('Choose event (war, fight etc.)')
    vari = form.vari
    preferred_name.append(vari)
    preferred_type.append('EVENT')
    options_type_filter.remove('EVENT')
  if 'ORG' in options_type_filter:
    found_type = 1
    form.vari = st.text_input('Choose organization...')
    vari = form.vari
    preferred_name.append(vari)
    preferred_type.append('ORG')
    options_type_filter.remove('ORG')
  for item in options_type_filter:
    if item != 'DATE':
      str_input = 'Choose ' + item.lower() + '...'
      form.vari = st.text_input(str_input)
      vari = form.vari
      preferred_name.append(vari)
      preferred_type.append(item)
  if preferred_name == ['']:
    preferred_name = []
    preferred_type = [] 
  return preferred_name,preferred_type

def choose_case(kw,options_type_filter, stats, eval, cross, preferred_name, preferred_type, num_select, form, json_object, tools, options_type, tools_list):
  if kw != '' and options_type_filter == [] and not stats and not eval and not cross:
   filter_sentences(kw,json_object, tools, options_type, tools_list)
  if (preferred_name != [] or form.d_start != '' or form.d_end != '') and not stats and not eval and not cross:
    filter_sentences_category(preferred_name, preferred_type,json_object, tools, options_type, options_type_filter, form.d_start,form.d_end)
  elif options_type_filter != [] and (preferred_name == [] and form.d_start == '' and form.d_end == '') and num_select != 0 and not stats and not eval and not cross:
    print_all_category(options_type_filter, json_object, options_type, tools, tools_list)

def print_annotate_sentence_by_index(num, json_object, tools, num_select, stats, eval, cross, kw, options_type, options_type_filter):
  str_ = make_tools_list(json_object)[0] + '-' + str(num)
  sentence = json_object[str_][0]['sentence']
  keys,values = make_keys_values(tools, json_object, num)

  if options_type_filter == [] and num_select != 0 and not stats and not eval and not cross and kw == '':
    annotate_text(sentence, keys, values, 'NOT', options_type)
    print_statistics_specific(json_object, num)
    with st.expander("Need information on a mentioned entity?"):
      give_info_entity(num,json_object)
