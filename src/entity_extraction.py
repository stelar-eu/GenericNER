
from backend_functions import *
from llm_foodNER_functions import *
from foodroberta_functions import *
from utils_init import *
import os

def entity_extraction(df, extraction_type, model, output_file, N = 10,
                           syntactic_analysis_tool = 'stanza', prompt_id = 0, ontology = None):
  ##Food NER
  df_scores = make_df_scores(set_foods = set(), set_no_foods = set())
  df_scores = update_time(df_scores)
  nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,ner')
  nlp_sp = spacy.load("en_core_web_sm")
  dict_metrics, dictionary = {}, {}
  syntactic = prompt_id == 0
  if syntactic_analysis_tool == 'stanza':
      chosen_nlp = nlp
  elif syntactic_analysis_tool == 'spacy':
      chosen_nlp = nlp_sp
  else:
      print('ERROR: NLP tool not available. Available tools: stanza, spacy')
      return '', {}
  texts_by_period, tags_by_period = make_lists_period_split(df,N)
  if 'mistral:7b' in model or 'llama2:7b' in model or 'openhermes:7b-v2.5' in model:
      texts_by_period_llm = clean_text(texts_by_period)
  prompt = 'Classify the following item as EDIBLE or NON EDIBLE. Desired format: [EDIBLE/NON EDIBLE]. Input:'
  prompt1 = 'Print only one comma-separated list of the foods, drinks or edible ingredients mentioned in the previous text. Do write a very short answer, with no details, just the list. If there are no foods, drinks or edible ingredients mentioned, print no.'
  prompt2 = 'You are a food allergy specialist and your task is to find anything edible, i.e. food, drink or ingredient, mentioned in the previous text. If you lose any edible item mentioned, there is a risk of someone getting allergy and you will be penalized. Print the edible items you found in a comma-separated list, each edible item printed separately and without further information. If there are no edible items mentioned in the text, print no.'
  prompt3 = 'Find any foods, drinks or edible ingredients mentioned in the previous text. Print them in a comma-separated list. If there are none, print no. Write a short answer.'
  prompts = [prompt,prompt1,prompt2,prompt3]
  chosen_prompt = prompts[prompt_id]
  df_scores_if,df_scores_sf,df_scores_ll = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
  new_df = pd.DataFrame()
  if 'food' in extraction_type and 'instafoodroberta' in model:
      df_scores_if = annotate_entities_foodroberta(df = df,df_scores = df_scores,texts_by_period = texts_by_period,true_tags = tags_by_period, start_text = 0, end_text = len(texts_by_period))
      if 'tags' in list(df.columns):
        dict_metrics = df_scores_to_dict(df_scores_if)
      all_entities = df_scores_if.loc['food_entities_total',:].tolist()[0]
      all_entities_merged = [item for sublist in all_entities for item in sublist]
      lst = annotate_text_tokens(df,all_entities_merged,N)
      dictionary = list_to_dict(lst,df[:N],'instafoodroberta',dictionary)
      new_df = food_data_to_csv(df[:N],all_entities_merged,'instafoodroberta',new_df)
  if 'food' in extraction_type and 'scifoodner' in model:
      df_scores_sf = annotate_entities_scifoodner(df = df, df_scores = df_scores, texts_by_period = texts_by_period, tags_by_period = tags_by_period, start_text = 0, end_text = len(texts_by_period))
      if 'tags' in list(df.columns):
        dict_metrics = df_scores_to_dict(df_scores)
      all_entities = df_scores_sf.loc['food_entities_total',:].tolist()[0]
      all_entities_merged = all_entities
      lst = annotate_text_tokens(df,all_entities_merged,N)
      dictionary = list_to_dict(lst,df[:N],'scifoodner',dictionary)
      new_df = food_data_to_csv(df[:N],all_entities_merged,'scifoodner',new_df)
  if ('food' in extraction_type) and ('mistral:7b' in model or 'llama2:7b' in model or 'openhermes:7b-v2.5' in model):
      model = model[1:-1].split()
      llm_models = [item for item in model if item in ['mistral:7b','llama2:7b','openhermes:7b-v2.5']]
      no_repetitions = len(llm_models)
      df_scores_ll = LLM_foodNER(df, df_scores, texts = texts_by_period_llm, true_tags = tags_by_period, start_text = 0, end_text = len(texts_by_period), no_repetitions = no_repetitions, repetition_LLM = llm_models, prompt = chosen_prompt, syntactic = syntactic, nlp = chosen_nlp)      
      if 'tags' in list(df.columns):
        dict_metrics = df_scores_to_dict(df_scores)
      all_entities = df_scores.loc['food_entities_total',:].tolist()[0]
      all_entities_merged = [item for sublist in all_entities for item in sublist]
      lst = annotate_text_tokens(df,all_entities_merged,N)
      dictionary = list_to_dict(lst,df[:N],str(llm_models),dictionary)
      new_df = food_data_to_csv(df[:N],all_entities_merged,str(llm_models),new_df)

  ##Generic NER
  df = make_df_by_argument(df, text_column = 'text', ground_truth_column = 'tags',csv_delimiter = ',', minio='')
  if extraction_type == 'all':
    names_df = ["PERSON", "NORP", "FAC", "ORG","GPE", "LOC", "PRODUCT", "DATE","TIME","PERCENT","MONEY",\
        "QUANTITY","ORDINAL", "CARDINAL", "EVENT", "WORK_OF_ART", "LAW","LANGUAGE","MISC"]
  else:
    names_df = [item.upper() for item in extraction_type if item != 'food']
  #Specify number of texts to be counted
  N = choose_num(df,N)	#defaults to N, if dataset has fewer, N = length of dataset

  df_results_list = []
  tools = [item for item in model if item in ['spacy_roberta','spacy','flair','stanza']]
  tools = rename_tools(tools)
  nlps = load_import_models(tools)
  for tool in tools:
   if names_df != []:
    nlp_cur = nlps[0]
    nlps = nlps[1:]
    if tool == 'Flair':
      start,end,dictionary = find_named_entities_flair(dictionary,N,nlp_cur,df,names_df)
    elif tool == 'Stanza':
      start,end,dictionary = find_named_entities_stanza(dictionary,N,nlp_cur,df,names_df)
    else:
      start,end,dictionary = find_named_entities(dictionary,N,nlp_cur,df,tool,names_df)
    if 'named_entities' in df.columns:										 #Ground truth provided, so perform evaluation
      if tool == 'Flair':
        df_results, returned = evaluate_flair(N, names_df, nlp_cur, df, names_df)
      elif tool == 'Stanza':
        df_results, returned = evaluate_stanza(N, names_df, nlp_cur, df, names_df)  #Evaluate tool on dataset
      else:
        df_results, returned = evaluate_tool(N, names_df, nlp_cur, df, tool,names_df)
      df_results_list.append(returned)
      dict_metrics = measure_write_result(df_results, dict_metrics, tool, start, end) 	 #Print results after comparison with ground truth 
      print('dict metrics generic:',dict_metrics)
    else:
      if tool == 'Flair':
        df_results_list.append(return_bio_format_flair(N, nlp_cur, df,names_df))
      elif tool == 'Stanza':
        df_results_list.append(return_bio_format_stanza(N, nlp_cur, df, names_df))  #Evaluate tool on dataset
      else:
        df_results_list.append(return_bio_format(N, nlp_cur, df, tool, names_df))
  if df_results_list != []:
    total_results_df = merge_dfs_horizontally(df_results_list)
    dict_metrics = cross_results(total_results_df,N,names_df, dict_metrics,print_df=False)
    new_df = generic_data_to_csv(df[:N],total_results_df,new_df)
    compare_results(total_results_df,N,names_df)
  if ontology is not None:
    pass
  new_df.to_csv(output_file + '.csv',index = False)
  write_json_file(dictionary,output_file,dict_metrics) #write_json_file(dictionary,output_file,dict_metrics)
  return output_file, dict_metrics    

def main():
  extraction_type = ['food']
  dataset = 'incidents.csv'
  minio=None
  model = ['flair','spacy_roberta','instafoodroberta']
  if 'mistral:7b' in model or 'llama2:7b' in model or 'openhermes:7b-v2.5' in model:
    model_name = 'LLM'
  else:
    model_name = model

  df = prepare_dataset_new(dataset, text_column = 'description', ground_truth_column = 'tags', product_column = 'product', minio = minio)
  if df.empty:
    return -1

  output_file = str(model)[1:-1] + '_results_' + str(extraction_type)[1:-1]
  output_file_path, dict_metrics = entity_extraction(df, extraction_type = extraction_type, model = model,
                                                   output_file = output_file, N= 10, ontology = None)
  print('CSV output_file_path:', output_file_path + '.csv')
  print('JSON output_file_path:', output_file_path + '.json')
  print('evaluation dictionary:', dict_metrics)

if __name__ == "__main__":
  main()
