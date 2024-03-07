from backend_functions import *
from llm_foodNER_functions import *
from foodroberta_functions import *
from utils_init import *
import os

def food_entity_extraction(df, extraction_type, model, output_file, N = 10,
                           syntactic_analysis_tool = 'stanza', prompt_id = 0):
    df_scores = make_df_scores(set_foods = set(), set_no_foods = set())
    df_scores = update_time(df_scores)
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,ner')
    nlp_sp = spacy.load("en_core_web_sm")
    dict_metrics = {}
    syntactic = prompt_id == 0
    if syntactic_analysis_tool == 'stanza':
      chosen_nlp = nlp
    elif syntactic_analysis_tool == 'spacy':
      chosen_nlp = nlp_sp
    else:
      print('ERROR: NLP tool not available. Available tools: stanza, spacy')
      return '', {}
    texts_by_period, tags_by_period = make_lists_period_split(df,N)
    if model != 'instafoodroberta' and model != 'scifoodner':
      texts_by_period = clean_text(texts_by_period)
    prompt = 'Classify the following item as EDIBLE or NON EDIBLE. Desired format: [EDIBLE/NON EDIBLE]. Input:'
    prompt1 = 'Print only one comma-separated list of the foods, drinks or edible ingredients mentioned in the previous text. Do write a very short answer, with no details, just the list. If there are no foods, drinks or edible ingredients mentioned, print no.'
    prompt2 = 'You are a food allergy specialist and your task is to find anything edible, i.e. food, drink or ingredient, mentioned in the previous text. If you lose any edible item mentioned, there is a risk of someone getting allergy and you will be penalized. Print the edible items you found in a comma-separated list, each edible item printed separately and without further information. If there are no edible items mentioned in the text, print no.'
    prompt3 = 'Find any foods, drinks or edible ingredients mentioned in the previous text. Print them in a comma-separated list. If there are none, print no. Write a short answer.'
    prompts = [prompt,prompt1,prompt2,prompt3]
    chosen_prompt = prompts[prompt_id]
    
    if model == 'instafoodroberta':
      df_scores = annotate_entities_foodroberta(df = df,df_scores = df_scores,texts_by_period = texts_by_period,true_tags = tags_by_period, start_text = 0, end_text = len(texts_by_period))
    elif model == 'scifoodner':
      df_scores = annotate_entities_scifoodner(df = df, df_scores = df_scores, texts_by_period = texts_by_period, tags_by_period = tags_by_period, start_text = 0, end_text = len(texts_by_period))
    else: 
      model = model[1:-1].split()
      no_repetitions = len(model)
      df_scores = LLM_foodNER(df, df_scores, texts = texts_by_period, true_tags = tags_by_period, start_text = 0, end_text = len(texts_by_period), no_repetitions = no_repetitions, repetition_LLM = model, prompt = chosen_prompt, syntactic = syntactic, nlp = chosen_nlp)      
    if 'tags' in list(df.columns):
      dict_metrics = df_scores_to_dict(df_scores)
    all_entities = df_scores.loc['food_entities_total',:].tolist()[0]
    if model == 'scifoodner':
      all_entities_merged = all_entities
    else:
      all_entities_merged = [item for sublist in all_entities for item in sublist]
    all_entities_cur = all_entities_merged
    lst = []
    id = -1
    for text in df['text'][:N]:
        id += 1
        lst.append(all_entities_cur[:len(text.split())])
        all_entities_cur = all_entities_cur[len(text.split()):]
    dictionary = list_to_dict(lst,df)
    tool = 'llm' if type(model)==list else model
    new_df = food_data_to_csv(df[:N],all_entities_merged,tool)
    if not os.path.exists('results'):
      os.mkdir('results') 
    new_df.to_csv(output_file + '.csv',index = False)
    write_json_file(dictionary,output_file)
    return output_file, dict_metrics    


def generic_entity_extraction(df, extraction_type, model, output_file, N = 10):
  names_df = ["PERSON", "NORP", "FAC", "ORG","GPE", "LOC", "PRODUCT", "DATE","TIME","PERCENT","MONEY",\
        "QUANTITY","ORDINAL", "CARDINAL", "EVENT", "WORK_OF_ART", "LAW","LANGUAGE","MISC"]
  #Specify number of texts to be counted
  N = choose_num(df,N)	#defaults to N, if dataset has fewer, N = length of dataset

  nlps = load_import_models()
  if extraction_type != 'all':
    dictionary = {}
    dict_metrics = {}
  df_results_list = []

  for tool in ['spaCy + RoBERTa','spaCy','Flair','Stanza']:
    nlp_cur = nlps[0]
    nlps = nlps[1:]
    if tool == 'Flair':
      start,end,dictionary = find_named_entities_flair(dictionary,N,nlp_cur,df)
    elif tool == 'Stanza':
      start,end,dictionary = find_named_entities_stanza(dictionary,N,nlp_cur,df)
    else:
      start,end,dictionary = find_named_entities(dictionary,N,nlp_cur,df,tool)
    if 'named_entities' in df.columns:										 #Ground truth provided, so perform evaluation
      if tool == 'Flair':
        df_results, returned = evaluate_flair(N, names_df, nlp_cur, df)
      elif tool == 'Stanza':
        df_results, returned = evaluate_stanza(N, names_df, nlp_cur, df)  #Evaluate tool on dataset
      else:
        df_results, returned = evaluate_tool(N, names_df, nlp_cur, df, tool)
      df_results_list.append(returned)
      dict_metrics = measure_write_result(df_results, dict_metrics, tool, start, end) 	 #Print results after comparison with ground truth 
      print('dict metrics generic:',dict_metrics)
    else:
      if tool == 'Flair':
        df_results_list.append(return_bio_format_flair(N, nlp_cur, df))
      elif tool == 'Stanza':
        df_results_list.append(return_bio_format_stanza(N, nlp_cur, df))  #Evaluate tool on dataset
      else:
        df_results_list.append(return_bio_format(N, nlp_cur, df, tool))

  total_results_df = merge_dfs_horizontally(df_results_list)
  dict_metrics = cross_results(total_results_df,N,names_df, dict_metrics)
  if extraction_type == 'generic':
    new_df = pd.DataFrame()
  generic_data_to_csv(df[:N],total_results_df,new_df)
  compare_results(total_results_df,N,names_df)
  new_df.to_csv(output_file + '.csv',index = False)
  write_json_file(dictionary,output_file)
  return output_file, dict_metrics    

def entity_extraction(df, extraction_type, model, output_file, N = 10,
                      syntactic_analysis_tool = 'stanza', prompt_id = 1):
  if extraction_type == 'food':
      output_file, dict_metrics  = food_entity_extraction(df, extraction_type, model, 
                                                          output_file, N = N,
                                                          syntactic_analysis_tool = syntactic_analysis_tool,
                                                          prompt_id = prompt_id)
  elif extraction_type == 'generic':
      output_file, dict_metrics  = generic_entity_extraction(df, extraction_type, model, 
                                                             output_file, N = N)

  return output_file, dict_metrics    


def main():
  extraction_type = 'food'
  dataset = 'foodbase.csv'
  minio=None
  model = 'instafoodroberta'
  if '[' in model:
    model_name = 'LLM'
  else:
    model_name = model
  if extraction_type == 'generic':
      df = make_df_by_argument(dataset, text_column = 'description', ground_truth_column = 'generic_tags', 
                               csv_delimiter = ',', minio=minio)
  elif extraction_type == 'food':
      df = prepare_dataset_new(dataset, text_column = 'text', ground_truth_column = 'iob_tags', minio = minio)
      if df.empty:
        return -1

  output_file = 'results/instafoodroberta_results'
  output_file_path, dict_metrics = entity_extraction(df, extraction_type = extraction_type, model = model,
                                                   output_file = output_file, N= 100)
  print('CSV output_file_path:', output_file_path + '.csv')
  print('JSON output_file_path:', output_file_path + '.json')
  print('evaluation dictionary:', dict_metrics)

if __name__ == "__main__":
  main()
