from backend_functions import *
from llm_foodNER_functions import *
from foodroberta_functions import *
from utils_init import *
import os

def entity_extraction(df, prediction_values, N = 10, output_file = 'ee_output',
                           syntactic_analysis_tool = 'stanza', split_by = 'period', prompt_id = 1, ontology = None):
  ##Food NER
  df_scores = make_df_scores(set_foods = set(), set_no_foods = set())
  df_scores = update_time(df_scores)
  dict_metrics, dictionary = {}, {}
  syntactic = prompt_id == 0
  if syntactic:
   nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,ner')
   nlp_sp = spacy.load("en_core_web_sm") 
   if syntactic_analysis_tool == 'stanza':
       chosen_nlp = nlp
   elif syntactic_analysis_tool == 'spacy':
       chosen_nlp = nlp_sp
   else:
       print('ERROR: NLP tool not available. Available tools: stanza, spacy')
       return '', {}
  else:
    chosen_nlp = None
  if split_by == 'period':
   texts_by_period, tags_by_period = make_lists_period_split(df,N)
  else:
   texts_by_period, tags_by_period = list(df['text'][:N]),list(df['tags'][:N])
  if 'food' in list(prediction_values.keys()) and (('mistral:7b' in prediction_values['food']) or ('llama2:7b' in prediction_values['food']) or ('openhermes:7b-v2.5' in prediction_values['food'])):
      texts_by_period_llm = clean_text(texts_by_period)
  prompt = 'Classify the following item as EDIBLE or NON EDIBLE. Desired format: [EDIBLE/NON EDIBLE]. Input:'
  prompt1 = 'Print only one comma-separated list of the foods, drinks or edible ingredients mentioned in the previous text. Do write a very short answer, with no details, just the list. If there are no foods, drinks or edible ingredients mentioned, print no.'
  prompt2 = 'You are a food allergy specialist and your task is to find anything edible, i.e. food, drink or ingredient, mentioned in the previous text. If you lose any edible item mentioned, there is a risk of someone getting allergy and you will be penalized. Print the edible items you found in a comma-separated list, each edible item printed separately and without further information. If there are no edible items mentioned in the text, print no.'
  prompt3 = 'Find any foods, drinks or edible ingredients mentioned in the previous text. Print them in a comma-separated list. If there are none, print no. Write a short answer.'
  prompts = [prompt,prompt1,prompt2,prompt3]
  chosen_prompt = prompts[prompt_id]
  df_scores_if,df_scores_sf,df_scores_ll = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
  new_df = pd.DataFrame()
  if 'food' in list(prediction_values.keys()) and 'instafoodroberta' in prediction_values['food']:
      df_scores_if = annotate_entities_foodroberta(df = df,df_scores = df_scores,texts_by_period = texts_by_period,true_tags = tags_by_period, start_text = 0, end_text = len(texts_by_period))
      all_entities = df_scores_if.loc['food_entities_total',:].tolist()[0]
      all_entities_merged = [item for sublist in all_entities for item in sublist]
      lst = annotate_text_tokens(df,all_entities_merged,N)
      dictionary = list_to_dict(lst,df[:N],'instafoodroberta',dictionary)
      new_df = food_data_to_csv(df[:N],all_entities_merged,'instafoodroberta',new_df,discard_non_entities = True)
      if 'tags' in list(df.columns):
        dict_metrics,dictionary = df_scores_to_dict(df_scores_if,dictionary,'instafoodroberta')
  if 'food' in list(prediction_values.keys()) and 'scifoodner' in prediction_values['food']:
      df_scores_sf = annotate_entities_scifoodner(df = df, df_scores = df_scores, texts_by_period = texts_by_period, tags_by_period = tags_by_period, start_text = 0, end_text = len(texts_by_period))
      all_entities = df_scores_sf.loc['food_entities_total',:].tolist()[0]
      all_entities_merged = all_entities
      lst = annotate_text_tokens(df,all_entities_merged,N)
      dictionary = list_to_dict(lst,df[:N],'scifoodner',dictionary)
      new_df = food_data_to_csv(df[:N],all_entities_merged,'scifoodner',new_df,discard_non_entities = True)
      if 'tags' in list(df.columns):
        dict_metrics,dictionary = df_scores_to_dict(df_scores_sf,dictionary,'scifoodner')
  if ('food' in list(prediction_values.keys())) and ('mistral:7b' in prediction_values['food'] or 'llama2:7b' in prediction_values['food'] or 'openhermes:7b-v2.5' in prediction_values['food']):
      llm_models = [item for item in prediction_values['food'] if item in ['mistral:7b','llama2:7b','openhermes:7b-v2.5']]
      no_repetitions = len(llm_models)
      df_scores_ll = LLM_foodNER(df, df_scores, texts = texts_by_period_llm, true_tags = tags_by_period, start_text = 0, end_text = len(texts_by_period), no_repetitions = no_repetitions, repetition_LLM = llm_models, prompt = chosen_prompt, syntactic = syntactic, nlp = chosen_nlp)
      all_entities = df_scores_ll.loc['food_entities_total',:].tolist()[0]
      all_entities_merged = [item for sublist in all_entities for item in sublist]
      lst = annotate_text_tokens(df,all_entities_merged,N)
      dictionary = list_to_dict(lst,df[:N],str(llm_models)[1:-1],dictionary)
      new_df = food_data_to_csv(df[:N],all_entities_merged,str(llm_models),new_df,discard_non_entities = True)
      if 'tags' in list(df.columns):
        dict_metrics,dictionary = df_scores_to_dict(df_scores_ll,dictionary,str(llm_models)[1:-1])

  ##Generic NER
  df = make_df_by_argument(df, text_column = 'text', ground_truth_column = 'tags',csv_delimiter = ',', minio='')
  #Specify number of texts to be counted
  N = choose_num(df,N)	#defaults to N, if dataset has fewer, N = length of dataset

  df_results_list = []
  tools = [item for sublist in list(prediction_values.values()) for item in sublist ]
  tools = set(tools)
  tools = [item for item in tools if item in ['spacy_roberta','spacy','flair','stanza']]
  tools = rename_tools(tools)
  if tools != []:
   nlps = load_import_models(tools)
  names_df_flair, names_df_stanza, names_df_spacy, names_df_spacy_rob = [],[],[],[]
  for tool in tools:
    nlp_cur = nlps[0]
    nlps = nlps[1:]
    if tool == 'Flair':
      names_df = [k for k, v in prediction_values.items() if 'flair' in v]
      names_df_flair = [item.upper() for item in names_df]
      start,end,dictionary = find_named_entities_flair(dictionary,N,nlp_cur,df,names_df_flair)
    elif tool == 'Stanza':
      names_df = [k for k, v in prediction_values.items() if 'stanza' in v]
      names_df_stanza = [item.upper() for item in names_df]
      start,end,dictionary = find_named_entities_stanza(dictionary,N,nlp_cur,df,names_df_stanza)
    else:
      names_df = [k for k, v in prediction_values.items() if 'spacy' in v]
      names_df_spacy = [item.upper() for item in names_df]
      names_df = [k for k, v in prediction_values.items() if 'spacy_roberta' in v]
      names_df_spacy_rob = [item.upper() for item in names_df]
      if tool == 'spaCy':
       start,end,dictionary = find_named_entities(dictionary,N,nlp_cur,df,tool,names_df_spacy)
      else:
       start,end,dictionary = find_named_entities(dictionary,N,nlp_cur,df,tool,names_df_spacy_rob)
    if 'named_entities' in df.columns:										 #Ground truth provided, so perform evaluation
      if tool == 'Flair':
        df_results, returned = evaluate_flair(N, names_df, nlp_cur, df, names_df_flair)
      elif tool == 'Stanza':
        df_results, returned = evaluate_stanza(N, names_df, nlp_cur, df, names_df_stanza)  #Evaluate tool on dataset
      elif tool == 'spaCy':
        df_results, returned = evaluate_tool(N, names_df, nlp_cur, df, tool,names_df_spacy)
      else:
        df_results, returned = evaluate_tool(N, names_df, nlp_cur, df, tool,names_df_spacy_rob)
      df_results_list.append(returned)
      dict_metrics = measure_write_result(df_results, dict_metrics, tool, start, end) 	 #Print results after comparison with ground truth 
      print('dict metrics generic:',dict_metrics)
    else:
      if tool == 'Flair':
        df_results_list.append(return_bio_format_flair(N, nlp_cur, df,names_df_flair))
      elif tool == 'Stanza':
        df_results_list.append(return_bio_format_stanza(N, nlp_cur, df, names_df_stanza))  #Evaluate tool on dataset
      elif tool == 'spaCy':
        df_results_list.append(return_bio_format(N, nlp_cur, df, tool, names_df_spacy))
      else:
        df_results_list.append(return_bio_format(N, nlp_cur, df, tool, names_df_spacy_rob))
  if df_results_list != []:
    total_results_df = merge_dfs_horizontally(df_results_list)
    names_df = list(set(names_df_flair + names_df_stanza + names_df_spacy + names_df_spacy_rob))
    dict_metrics = cross_results(total_results_df,N,names_df,prediction_values,dict_metrics,print_df=False)
    new_df = generic_data_to_csv(df[:N],total_results_df,new_df,discard_non_entities = True)
    compare_results(total_results_df,N,names_df)
  if ontology is not None:
    pass
  new_df.to_csv(output_file + '.csv',index = False)
  write_json_file(dictionary,output_file,dict_metrics)
  return output_file, dict_metrics

def main():
  minio=None
  dataset, text_column, ground_truth_column, product_column, csv_delimiter, prediction_values, N, ontology, minio = read_configuration_file('../config_file.ini')
  df = prepare_dataset_new(dataset, text_column = text_column, ground_truth_column = ground_truth_column, product_column = product_column, csv_delimiter = csv_delimiter, minio = minio)
  if df.empty:
    return -1

  output_file = generate_output_file_name(dataset,prediction_values)
  output_file_path, dict_metrics = entity_extraction(df, prediction_values = prediction_values,
                                                   output_file = output_file, N= N, ontology = None)
  print('CSV output_file_path:', output_file_path + '.csv')
  print('JSON output_file_path:', output_file_path + '.json')
  print('evaluation dictionary:', dict_metrics)

if __name__ == "__main__":
  main()
