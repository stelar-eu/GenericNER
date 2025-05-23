import re
import time
from llm_foodNER_functions import *
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def convert_entities_to_list(text, entities):
        ents = []
        for ent in entities:
            e = {"start": ent["start"], "end": ent["end"], "label": ent["entity_group"]}
            if ents and -1 <= ent["start"] - ents[-1]["end"] <= 1 and ents[-1]["label"] == e["label"]:
                ents[-1]["end"] = e["end"]
                continue
            ents.append(e)

        return [text[e["start"]:e["end"]] for e in ents]

def prepare_instafoodroberta():
  tokenizer = AutoTokenizer.from_pretrained("Dizex/InstaFoodRoBERTa-NER") 
  model = AutoModelForTokenClassification.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
  pipe_food = pipeline("ner", model=model, tokenizer=tokenizer)
  return pipe_food

def annotate_entities_foodroberta(df, df_scores, texts_by_period, true_tags, start_text, end_text):
  i = -1
  pipe_food = prepare_instafoodroberta()
  hits, discrete_hits, discrete_hits_prec, partial_hits, partial_hits_prec, wrong_constructions, all_entities, all_entities_prec, error_sentences, food_entities_total, predicted_entities_total, missed_entities, fp_entities, start, set_foods, set_no_foods = get_info_from_df(df_scores)
  for i in range(len(texts_by_period)):
    if i <= start_text - 1:
      continue
    example = texts_by_period[i]
    ner_entity_results = pipe_food(example, aggregation_strategy="simple")
    if ner_entity_results != []:
      entities_list = convert_entities_to_list(example, ner_entity_results)
      entities_list = list(dict.fromkeys(entities_list))
    else:
      entities_list = []
    food_entities = find_entities_in_text(entities_list,predicted_entities_total,food_entities_total, example, print_df = False)

    if 'tags' in df.columns:
      hits, discrete_hits, partial_hits, all_entities, wrong_constructions, missed_entities = evaluate_results(food_entities, true_tags[i], hits, discrete_hits, partial_hits, all_entities, wrong_constructions,missed_entities, example)
      hits_trash, discrete_hits_prec, partial_hits_prec, all_entities_prec, wrong_constructions_trash, fp_entities = evaluate_results(true_tags[i], food_entities, hits, discrete_hits_prec, partial_hits_prec, all_entities_prec, wrong_constructions, fp_entities, example)
    if i >= end_text:
       break
  end = time.time()
  print('time elapsed (InstaFoodRoBERTa): {0:.2f}'.format(end-start), 'seconds')
  df_scores = update_df_scores(df_scores, hits, discrete_hits, discrete_hits_prec, partial_hits, partial_hits_prec, wrong_constructions, all_entities, all_entities_prec, error_sentences, food_entities_total, predicted_entities_total, missed_entities, fp_entities, set_foods, set_no_foods)
  return df_scores
