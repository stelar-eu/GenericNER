import numpy as np
import pandas as pd
import torch
from simpletransformers.ner import NERModel
import shutil
from sklearn.metrics import classification_report
from llm_foodNER_functions import *

bert_model_config={'biobert':'dmis-lab/biobert-v1.1', 'roberta': 'roberta-base', 'bert':'bert-base-uncased', 'scibert':'allenai/scibert_scivocab_uncased'}

def evaluate_model(texts, tags, bert_model_name, start_text, end_text):
    ner_model = NERModel(
        bert_model_name if bert_model_name=='roberta' else 'bert',
        'cafeteria/best',
        args={"max_seq_length": 512}
    )
    results_list = save_predictions(texts[start_text:end_text], tags[start_text:end_text], ner_model = ner_model)
    return results_list


def save_predictions(texts, tags, ner_model):
    sum = 0
    texts_list = []
    inner_list = []
    predictions = pd.DataFrame()
    split_num = 5
    texts = clean_text(texts)
    for text in texts:
      for token in text.split():
        if len(token) > 100:
          token = token[:int(len(token)/3)]
        inner_list.append(token)
        sum += 1
        if '.' in token or token == '.' or len(inner_list) > 100:
          if len(inner_list) > 100:
            inner_list[-1] += '.'
          texts_list.append(inner_list)
          inner_list = []
    texts_list.append(inner_list)
    for part in range(split_num):
      if part == split_num - 1:
        preds, model_outputs=ner_model.predict(texts_list[part*(int(len(texts_list)/split_num)):len(texts_list)], split_on_space=False)
      else:
        preds, model_outputs=ner_model.predict(texts_list[part*(int(len(texts_list)/split_num)):(part+1)*(int(len(texts_list)/split_num))], split_on_space=False)
      df_preds = pd.DataFrame(preds)
      predictions=pd.concat([predictions,pd.DataFrame([ list(p.items())[0] for sentence_preds in preds for p in sentence_preds])], ignore_index = True)
    predictions.to_csv('predictions.csv')
    predictions.columns=['words','predictions']
    for i in range(len(predictions['predictions'])):
        if predictions['predictions'][i] != 'O':
            predictions['predictions'][i] += '-FOOD'

    if tags != []:
        tags_flattened = [tag for sublist in tags for tag in sublist]
        report = classification_report(predictions['predictions'],tags_flattened, output_dict=True)
        report = pd.DataFrame(report)
        print(report)
    return list(predictions['predictions'])
