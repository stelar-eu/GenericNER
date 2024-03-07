import streamlit as st
import pandas as pd
import csv
import sys
import json
import re
import datetime
from annotated_text import annotated_text
from configparser import ConfigParser
from entity_extraction_functions import *
from streamlit.components.v1 import html
from io import StringIO

st.sidebar.success("Welcome!")	#welcome message

json_object = make_json_object()	#read and load json file. Defaults to 'data.json' in current folder

tools_list = make_tools_list(json_object)	#make tools list from tools found in json file
n_tools = len(tools_list)
tools_list = ['All'] + tools_list	#add 'All' option

list_categories = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']

stats = st.sidebar.button('Global dataset statistics')		#print statistics
if stats:
  print_statistics(json_object, list_categories, tools_list)

eval = st.sidebar.button('Evaluation')		#print evaluation metrics if ground truth given
if eval:
  print_evaluation(json_object, list_categories, tools_list)

cross = st.sidebar.button('Compare results')		#compare results of tools
if cross:
  print_cross(json_object, list_categories, tools_list)

form = st.form('Manual assign')		#form used in expander

preferred_name = []		#used in filtering by category
preferred_type = []
kw = ''			#used in filtering by keyword

with st.sidebar.expander('Choose sentence'):
  num,num_select = select_index(form, json_object, n_tools)	#index of sentence to annotate and print
  options_type_filter = select_options_type_filter(form, list_categories)     #select category to filter sentences
  
  if 'DATE' in options_type_filter:		#case: date is chosen as filter
    form.d_start = st.text_input("From")
    form.d_end = st.text_input("To")
  else:
    form.d_start = ''
    form.d_end = ''
  preferred_name, preferred_type = preferred_entity(options_type_filter,form)
  kw = select_keyword(form)	#select keyword for filtering

options_type = choose_entity_type(list_categories)	#entity type to annotate(ORG,LOC,ALL etc.)
tools = choose_tools(tools_list)  #tool type to annotate(ORG,LOC,ALL etc.)

#if filter is given, print filtered sentences. If not and keyword is given, print texts containing keyword. If no filter and no keyword is given, print sentence by index.
choose_case(kw,options_type_filter, stats, eval, cross, preferred_name, preferred_type, num_select, form, json_object, tools, options_type, tools_list)		
print_annotate_sentence_by_index(num, json_object, tools, num_select, stats, eval, cross, kw, options_type, options_type_filter)

