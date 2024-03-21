import os
import csv
import pandas as pd
from simpletransformers.language_representation import RepresentationModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, train_test_split
from utils_train import evaluate_model
from llm_foodNER_functions import *

def annotate_entities_scifoodner(df, df_scores, texts_by_period, tags_by_period, start_text, end_text):
    hits, discrete_hits, discrete_hits_prec, partial_hits, partial_hits_prec, wrong_constructions, all_entities, all_entities_prec, error_sentences, food_entities_total, predicted_entities_total, missed_entities, fp_entities, start, set_foods, set_no_foods = get_info_from_df(df_scores)
    food_entities_flattened = evaluate_model(texts_by_period, tags_by_period, bert_model_name = 'biobert', start_text = start_text, end_text = end_text)
    food_entities_total = food_entities_flattened
    if 'tags' in df.columns:
      tags_flattened = [item_int for item in tags_by_period for item_int in item]
      hits, discrete_hits, partial_hits, all_entities, wrong_constructions, missed_entities = evaluate_results(food_entities_total, tags_flattened, hits, discrete_hits, partial_hits, all_entities, wrong_constructions,missed_entities, '')
      hits_trash, discrete_hits_prec, partial_hits_prec, all_entities_prec, wrong_constructions_trash, fp_entities = evaluate_results(tags_flattened, food_entities_total, hits, discrete_hits_prec, partial_hits_prec, all_entities_prec, wrong_constructions, fp_entities, '')
      print(discrete_hits,'correct', partial_hits, 'partial')
    if 'tags' in df.columns:
      evaluation_metrics(discrete_hits_prec,all_entities_prec,discrete_hits,all_entities,partial_hits,partial_hits_prec,wrong_constructions,error_sentences, to_dict = False)
    end = time.time()
    print('time elapsed (SciFoodNER): {0:.2f}'.format(end-start), 'seconds') 
    df_scores = update_df_scores(df_scores, hits, discrete_hits, discrete_hits_prec, partial_hits, partial_hits_prec, wrong_constructions, all_entities, all_entities_prec, error_sentences, food_entities_total, predicted_entities_total, missed_entities, fp_entities, set_foods, set_no_foods)
    return df_scores
