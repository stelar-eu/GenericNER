import regex
import pandas as pd
import os
import re
import time
import stanza
import spacy
import json
import matplotlib.pyplot as plt
from minio import Minio
from configparser import ConfigParser

def prepare_dataset_new(ds_path, text_column, ground_truth_column, product_column, csv_delimiter, minio):
    if ds_path.startswith('s3://'):
        bucket, key = ds_path.replace('s3://', '').split('/', 1)
        client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
        df = pd.read_csv(client.get_object(bucket, key), delimiter = csv_delimiter)
    else:
        df = pd.read_csv(ds_path,delimiter = csv_delimiter)
    if text_column not in list(df.columns):
        print('Error: no column named', text_column, '.')
        return pd.DataFrame()
    if ground_truth_column in list(df.columns):
        df[ground_truth_column] = df[ground_truth_column].apply(lambda x: x[2:-2].split('\', \''))
        df = df[[text_column,ground_truth_column,product_column]] if product_column in list(df.columns) else df[[text_column,ground_truth_column]]
        for i in range(len(df)):
          if len(df[ground_truth_column][i]) != len(df[text_column][i].split()):
            print('Error: the number of tags must be the same as the number of tokens in every text.')
            return pd.DataFrame()
    else:
        df = df[[text_column,product_column]] if product_column in list(df.columns) else df[[text_column]]
    df = df.rename(columns={text_column: "text", ground_truth_column:"tags", product_column:"product"})
    df = df.drop_duplicates(subset = 'text',ignore_index = True)
    return df

def prepare_dataset(ds_path, text_column, ground_truth_column, minio):
    if ds_path.startswith('s3://'):
        bucket, key = ds_path.replace('s3://', '').split('/', 1)
        client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
        df = pd.read_csv(client.get_object(bucket, key))
    else:
        df = pd.read_csv(ds_path)

    if len(list(df.columns)) == 2 and ground_truth_column in list(df.columns)[1]:
      df[ground_truth_column] = df[ground_truth_column].apply(lambda x: x[2:-2].split('\', \''))
    df = df.rename(columns={text_column: "text", ground_truth_column:"tags"})
    for i in range(len(df)):
        if len(df['tags'][i]) != len(df['text'][i].split()):
          print('Error: the number of tags must be the same as the number of tokens in every text.')
          df = pd.DataFrame()
    return df

def prepare_dataset_ak(ds_path, text_column, preprocess, minio):
    if ds_path.startswith('s3://'):
        bucket, key = ds_path.replace('s3://', '').split('/', 1)
        client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
        df = pd.read_csv(client.get_object(bucket, key))
    else:
        df = pd.read_csv(ds_path)
    
    df = df[[text_column]]
    df = df.drop_duplicates()
    df = df.reset_index(drop = True)
    df = df.rename(columns = {text_column:'text'})
    if preprocess:
        i = -1
        for test_string in df['text']:
            i += 1
            test_string = test_string.replace('\n', ' ')
            test_string = test_string.replace('\r', ' ')
            test_string = test_string.replace('   ', ' ')
            df.loc[i,'text'] = test_string
    return df

def prepare_dataset_scientific(path):
    df = pd.read_csv(path)
    df = df.drop(columns = df.columns[0], axis = 1)
    df['iob_tags'] = df['iob_tags'].apply(lambda x: x[2:-2].split('\', \''))
    
    for i in range(len(df)):
        while '\"' in df.loc[i,'text']:
            old_len = (len(df.loc[i,'text'].split()))
            df.loc[i,'text'] = df.loc[i,'text'][:df.loc[i,'text'].index('\"')] + ',' + df.loc[i,'text'][df.loc[i,'text'].index('\"')+1:]
        while ' \' ' in df.loc[i,'text']:
            old_len = (len(df.loc[i,'text'].split()))
            df.loc[i,'text'] = df.loc[i,'text'][:df.loc[i,'text'].index('\'')] + ',' + df.loc[i,'text'][df.loc[i,'text'].index('\'')+1:]
        while '\'' in df.loc[i,'text']:
            old_len = (len(df.loc[i,'text'].split()))
            df.loc[i,'text'] = df.loc[i,'text'][:df.loc[i,'text'].index('\'')] + df.loc[i,'text'][df.loc[i,'text'].index('\'')+1:]
    return df
    
def ask_ollama(text, LLM, prompt = False):
    if not prompt:
        text = '<<'+ text + '>>. Are there any edible products mentioned in the previous text? If no, print no. If yes, print the foods and drinks mentioned in a comma-separated list. Provide only the comma-separated list.'
        
    else:
        text = '<<' + text + '>>. ' + prompt
    list_llms_avail = ['mistral:7b','mistral:7b-instruct','openhermes:7b-v2.5','orca2:7b','llama2:7b']
    if LLM not in list_llms_avail:
        print('LLM not available. Choose one of the following: ', list_llms_avail)
        return -1
    str_for_ollama = "\'{\n\"model\":\"" + LLM + "\",\n\"prompt\":\"" + text + "\"\n}\'"
    try:
      bashCommand = "curl http://localhost:11434/api/generate -d " + str_for_ollama + ' > out.txt'
      os.system(bashCommand)
      answer = ''
      with open("out.txt", "r") as txt_file:
        stri = txt_file.read()
      for line in stri.split('\n'):
        try:
          content = line[line.index("response\":") + 11:line.index("\",\"done")]
          if content == '\\n':
              answer += '\n'
          else: 
              answer += line[line.index("response\":") + 11:line.index("\",\"done")]
        except:
          1
    except:
      print('error')
      no_response = 1
      return '', no_response
    no_response = 0
    return answer, no_response

def ask_ollama_syntactic(entities_list, LLM, prompt = False):
    text = 'Print the following list: ' 
    for item in entities_list:
        text += item + ': , '
    text = text[:-2]
    text += ' with the non-edible items removed. If nothing is left, print no. Print only the list and nothing more.'
    if prompt:
        text = prompt
    if type(entities_list) == list:
        for item in entities_list:
            text += item + ': , '
    else:
        text += entities_list + ': , '
    text = text[:-2]    
    list_llms_avail = ['mistral:7b','mistral:7b-instruct','openhermes:7b-v2.5','orca2:7b','llama2:7b']
    if LLM not in list_llms_avail:
        print('LLM not available. Choose one of the following: ', list_llms_avail)
        return -1
    str_for_ollama = "\'{\n\"model\":\"" + LLM + "\",\n\"prompt\":\"" + text + "\"\n}\'"
    try:
      bashCommand = "curl http://localhost:11434/api/generate -d " + str_for_ollama + ' > out.txt'
      os.system(bashCommand)
      answer = ''
      with open("out.txt", "r") as txt_file:
        stri = txt_file.read()
      for line in stri.split('\n'):
        try:
          content = line[line.index("response\":") + 11:line.index("\",\"done")]
          if content == '\\n':
              answer += '\n'
          else: 
              answer += line[line.index("response\":") + 11:line.index("\",\"done")]
        except:
          1
    except:
      print('error')
      no_response = 1
      return '', no_response
    no_response = 0
    return answer, no_response
    
def ask_ollama_python(text):
    text = "The comma-separated list of the foods and drinks in text: I went to the grocery store to buy wine and chocolate is [wine,chocolate]. The comma-separated list of the foods and drinks mentioned in the following text: " + text + "is (If there are no food entities, print 'no'):"
    str_for_ollama = "\'{\n\"model\":\"openhermes:7b\",\n\"prompt\":\"" + text + "\"\n}\'"
    try:
      bashCommand = "curl http://localhost:11434/api/generate -d " + str_for_ollama + ' > out.txt'
      os.system(bashCommand)
      answer = ''
      with open("out.txt", "r") as txt_file:
        stri = txt_file.read()
      for line in stri.split('\n'):
        try:
          content = line[line.index("response\":") + 11:line.index("\",\"done")]
          if content == '\\n':
              answer += '\n'
          else: 
              answer += line[line.index("response\":") + 11:line.index("\",\"done")]
        except:
          1
    except:
      print('error')
      no_response = 1
      return '', no_response
    no_response = 0
    return answer, no_response

def ask_vicuna(question):
    try:
      completion = openai.ChatCompletion.create(
      model=model,
      messages=[{"role": "user", "content": question}]
      )
    except:  #too many tokens for vicuna
      no_response = 1
      return '', no_response
    answer = completion.choices[0].message.content
    no_response = 0
    return answer, no_response

def entities_by_orca(answer):
    entities_list = []
    if 'inal answer:' in answer:
        answer = answer[answer.index('inal answer:') + len('inal answer:') + 1:]
        if 'text are' in answer:
            answer = answer[answer.index('text are') + len('text are') + 1:]
    while answer[0] == '\n' or answer[0] == ' ':
        answer = answer[1:]
    if len(answer.split()) == 1 or ('\n' not in answer and ',' not in answer and 'and ' not in answer):
        entities_list.append(answer)
    for idx in range(len(answer)):
        try:
            if answer[idx].isnumeric():                            #limited
                if answer[idx+1] == '.' and answer[idx+2] == ' ':
                    answer = answer[:idx] + '*' + answer[idx+2:]
                if answer[idx+1].isnumeric() and answer[idx+2] == '.':
                    answer = answer[:idx] + '*' + answer[idx+3:]
            if answer[idx+1] == ' ' and answer[idx] == '-':
                answer = answer[:idx] + '*' + answer[idx+1:]
        except:
            1
    if answer.count('\n\n') >= 2:
        indices_endl = [i for i in range(len(answer)-1) if answer[i:i+2] == "\n\n"]
        answer = answer[indices_endl[-1]:]
    if (',' in answer or 'and ' in answer) and '*' not in answer:
        try:
            shift = -1
            for char in answer[answer.index(':'):]:
                shift += 1
                if char.isalpha():
                    break
            if ', ' in answer:
                for entity in re.split(', |and ',answer[answer.index(':') + shift:]):
                    entities_list.append(entity)
            else:
                for entity in re.split(',|and ',answer[answer.index(':') + shift:]):
                    entities_list.append(entity)
        except:
            if ', ' in answer:
              for entity in re.split(', |and ',answer):
                entities_list.append(entity)
            else:
              for entity in re.split(',|and ',answer):
                entities_list.append(entity)
    if '*' in answer:
        try:
            shift = -1
            for char in answer[answer.index(':'):]:
                shift += 1
                if char.isalpha() or char == '*':
                    if char == '*':
                        shift += 1
                    break
            for entity in answer[answer.index(':') + shift:].split('*'):
                if '\n' in entity:
                  entities_list.append(entity[1:-1])
                elif '\\n' in entity:
                  entities_list.append(entity[1:-2])
                else:
                  entities_list.append(entity[1:])
        except:
            for entity in answer.split('*'):
                if '\n' in entity:
                  entities_list.append(entity[1:-1])
                elif '\\n' in entity:
                  entities_list.append(entity[1:-2])
                else:
                  entities_list.append(entity[1:])
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    for food_entity in entities_list:
        if ':' in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(':')]
            food_entity = food_entity[:food_entity.index(':')]
        if '- ' in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('- ')]
            food_entity = food_entity[:food_entity.index('- ')]
        if food_entity.count('\"') == 1 and food_entity[-1] == '\"' and '(' not in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if '.' in food_entity and '(' not in food_entity:
          if food_entity.index('.') != len(food_entity) -1:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')] + food_entity[food_entity.index('.')+1:]
          else:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')]
        if len(food_entity) > 0 and food_entity[-1] == ')' and '(' not in food_entity:
            try:
                entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            except:
                print('exception')
        if '(' in food_entity:
            if ' (' in food_entity:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(' (')]
            else:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('(')]
    for food_entity in entities_list:
        if ', ' in food_entity or 'and ' in food_entity:
            for entity in re.split(', |and ',food_entity):
                    entities_list.append(entity)
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    to_remove = []
    for food_entity in entities_list:
        if food_entity[-1] == ' ':
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if '\"' in food_entity or '\\"' in food_entity:
            to_remove.append(food_entity)
    entities_list = [e for e in entities_list if e not in to_remove]
    return entities_list

def entities_one_item(answer):
    while answer[0] == '\n' or answer[0] == ' ':
        answer = answer[1:]
    if 'inal answer:' in answer:
        answer = answer[answer.index('inal answer:') + len('inal answer:') + 1:]
    if answer[:8] == '[edible]' or answer[:6] == 'edible':
        return 1
    else:
        return 0
    
def entities_by_llama2_syntactic(answer):
    entities_list = []
    if 'inal answer:' in answer:
        answer = answer[answer.index('inal answer:') + len('inal answer:') + 1:]
        if 'text are' in answer:
            answer = answer[answer.index('text are') + len('text are') + 1:]
    while answer[0] == '\n' or answer[0] == ' ':
        answer = answer[1:]
    if len(answer.split()) == 1 or ('\n' not in answer and ',' not in answer and 'and ' not in answer):
        entities_list.append(answer)
    for idx in range(len(answer)):
        try:
            if answer[idx].isnumeric():                            #limited
                if answer[idx+1] == '.' and answer[idx+2] == ' ':
                    answer = answer[:idx] + '*' + answer[idx+2:]
                if answer[idx+1].isnumeric() and answer[idx+2] == '.':
                    answer = answer[:idx] + '*' + answer[idx+3:]
            if answer[idx+1] == ' ' and answer[idx] == '-':
                answer = answer[:idx] + '*' + answer[idx+1:]
        except:
            1
    if answer.count('\n\n') >= 2: 
        indices_endl = [i for i in range(len(answer)-1) if answer[i:i+2] == "\n\n"]
        try:
            answer = answer[indices_endl[1]:indices_endl[2]]
        except:
            answer = answer[indices_endl[0]:indices_endl[1]]
    if (',' in answer or 'and ' in answer) and '*' not in answer:
        try:
            shift = -1
            for char in answer[answer.index(':'):]:
                shift += 1
                if char.isalpha():
                    break
            if ', ' in answer:
                for entity in re.split(', |and ',answer[answer.index(':') + shift:]):
                    entities_list.append(entity)
            else:
                for entity in re.split(',|and ',answer[answer.index(':') + shift:]):
                    entities_list.append(entity)
        except:
            if ', ' in answer:
              for entity in re.split(', |and ',answer):
                entities_list.append(entity)
            else:
              for entity in re.split(',|and ',answer):
                entities_list.append(entity)
    if '*' in answer:
        try:
            shift = -1
            for char in answer[answer.index(':'):]:
                shift += 1
                if char.isalpha() or char == '*':
                    if char == '*':
                        shift += 1
                    break
            for entity in answer[answer.index(':') + shift:].split('*'):
                if '\n' in entity:
                  entities_list.append(entity[1:-1])
                elif '\\n' in entity:
                  entities_list.append(entity[1:-2])
                else:
                  entities_list.append(entity[1:])
        except:
            for entity in answer.split('*'):
                if '\n' in entity:
                  entities_list.append(entity[1:-1])
                elif '\\n' in entity:
                  entities_list.append(entity[1:-2])
                else:
                  entities_list.append(entity[1:])
    if '\n' in answer and (',' not in answer) and '\n\n' not in answer:
        for item in answer.split('\n'):
            entities_list.append(item)
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    for food_entity in entities_list:
        if ':' in food_entity and 'no food' not in food_entity[food_entity.index(':'):] and 'not a food' not in food_entity[food_entity.index(':'):] and 'not food' not in food_entity[food_entity.index(':'):] and 'food' in food_entity[food_entity.index(':'):]:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(':')]
            food_entity = food_entity[:food_entity.index(':')]
        if '- ' in food_entity and 'no food' not in food_entity[food_entity.index('-'):] and 'not a food' not in food_entity[food_entity.index('-'):] and 'not food' not in food_entity[food_entity.index('-'):] and 'food' in food_entity[food_entity.index('-'):]:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('- ')]
            food_entity = food_entity[:food_entity.index('- ')]
        if food_entity.count('\"') == 1 and food_entity[-1] == '\"' and '(' not in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if '.' in food_entity and '(' not in food_entity:
          if food_entity.index('.') != len(food_entity) -1:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')] + food_entity[food_entity.index('.')+1:]
          else:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')]
        if len(food_entity) > 0 and food_entity[-1] == ')' and '(' not in food_entity:
            try:
                entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            except:
                print('exception')
        if '(' in food_entity and 'not a food' not in food_entity and 'not food' not in food_entity:
            if ' (' in food_entity:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(' (')]
            else:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('(')]
    for food_entity in entities_list:
        if ', ' in food_entity or 'and ' in food_entity:
            for entity in re.split(', |and ',food_entity):
                    entities_list.append(entity)
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    to_remove = []
    for food_entity in entities_list:
        if food_entity[-1] == ' ':
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if '\"' in food_entity or '\\"' in food_entity:
            to_remove.append(food_entity)
    entities_list = [e for e in entities_list if e not in to_remove]
    return entities_list

def entities_by_mistral_instruct_syntactic(answer):
    entities_list = []
    while answer[0] == '\n' or answer[0] == ' ':
        answer = answer[1:]
    if '\n' in answer and (',' not in answer) and '\n\n' not in answer:
        for item in answer.split('\n'):
            entities_list.append(item)
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    for food_entity in entities_list:
        if ':' in food_entity and 'no food' not in food_entity[food_entity.index(':'):] and 'not a food' not in food_entity[food_entity.index(':'):] and 'not food' not in food_entity[food_entity.index(':'):] and 'food' in food_entity[food_entity.index(':'):]:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(':')]
            food_entity = food_entity[:food_entity.index(':')]
        if '- ' in food_entity and 'no food' not in food_entity[food_entity.index('-'):] and 'not a food' not in food_entity[food_entity.index('-'):] and 'not food' not in food_entity[food_entity.index('-'):] and 'food' in food_entity[food_entity.index('-'):]:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('- ')]
            food_entity = food_entity[:food_entity.index('- ')]
        if food_entity.count('\"') == 1 and food_entity[-1] == '\"' and '(' not in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if '.' in food_entity and '(' not in food_entity:
          if food_entity.index('.') != len(food_entity) -1:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')] + food_entity[food_entity.index('.')+1:]
          else:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')]
        if len(food_entity) > 0 and food_entity[-1] == ')' and '(' not in food_entity:
            try:
                entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            except:
                print('exception')
        if '(' in food_entity and 'not a food' not in food_entity and 'not food' not in food_entity:
            if ' (' in food_entity:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(' (')]
            else:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('(')]
    for food_entity in entities_list:
        if ', ' in food_entity or 'and ' in food_entity:
            for entity in re.split(', |and ',food_entity):
                    entities_list.append(entity)
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    to_remove = []
    for food_entity in entities_list:
        if food_entity[-1] == ' ':
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if '\"' in food_entity or '\\"' in food_entity:
            to_remove.append(food_entity)
    entities_list = [e for e in entities_list if e not in to_remove]
    return entities_list
    
def entities_by_mistral(answer):
    entities_list = []
    if 'inal answer:' in answer:
        answer = answer[answer.index('inal answer:') + len('inal answer:') + 1:]
        if 'text are' in answer:
            answer = answer[answer.index('text are') + len('text are') + 1:]
    if 'utput:' in answer:
        answer = answer[answer.index('utput:') + len('utput:') + 1:]        
    while answer[0] == '\n' or answer[0] == ' ':
        answer = answer[1:]
    for idx in range(len(answer)):
        try:
            if answer[idx].isnumeric():                            #limited
                if answer[idx+1] == '.' and answer[idx+2] == ' ':
                    answer = answer[:idx] + '*' + answer[idx+2:]
                if answer[idx+1].isnumeric() and answer[idx+2] == '.':
                    answer = answer[:idx] + '*' + answer[idx+3:]
            if answer[idx+1] == ' ' and answer[idx] == '-':
                answer = answer[:idx] + '*' + answer[idx+1:]
        except:
            1
    if answer.count('\n\n') >= 2: 
        indices_endl = [i for i in range(len(answer)-1) if answer[i:i+2] == "\n\n"]
        answer = answer[:indices_endl[0]]
    if answer.count('\n\n') == 1:
        if ':' not in answer[:answer.index('\n')]:
            answer = answer[:answer.index('\n')]
    if len(answer.split()) == 1 or ('\n' not in answer and ',' not in answer and 'and ' not in answer):
        entities_list.append(answer)
    if (',' in answer or 'and ' in answer) and '*' not in answer:
        try:
            shift = -1
            for char in answer[answer.index(':'):]:
                shift += 1
                if char.isalpha():
                    break
            if ', ' in answer:
                for entity in re.split(', |and ',answer[answer.index(':') + shift:]):
                    entities_list.append(entity)
            else:
                for entity in re.split(',|and ',answer[answer.index(':') + shift:]):
                    entities_list.append(entity)
        except:
            if ', ' in answer:
              for entity in re.split(', |and ',answer):
                entities_list.append(entity)
            else:
              for entity in re.split(',|and ',answer):
                entities_list.append(entity)
    if '*' in answer:
        try:
            shift = -1
            for char in answer[answer.index(':'):]:
                shift += 1
                if char.isalpha() or char == '*':
                    if char == '*':
                        shift += 1
                    break
            for entity in answer[answer.index(':') + shift:].split('*'):
                if '\n' in entity:
                  entities_list.append(entity[1:-1])
                elif '\\n' in entity:
                  entities_list.append(entity[1:-2])
                else:
                  entities_list.append(entity[1:])
        except:
            for entity in answer.split('*'):
                if '\n' in entity:
                  entities_list.append(entity[1:-1])
                elif '\\n' in entity:
                  entities_list.append(entity[1:-2])
                else:
                  entities_list.append(entity[1:])
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    for food_entity in entities_list:
        if "\\\"." in food_entity and food_entity.count("\\\".") == 1:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index("\\\".")]
            food_entity = food_entity[:food_entity.index("\\\".")]
        if "\\\"" in food_entity and food_entity.count("\\\"") == 1:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index("\\\"")]
            food_entity = food_entity[:food_entity.index("\\\"")]
        if "\\\"" in food_entity and food_entity.count("\\\"") == 2:
            entities_list[entities_list.index(food_entity)] = food_entity[food_entity.index("\\\""):]
            food_entity = food_entity[food_entity.index("\\\""):]
        if ':' in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(':')]
            food_entity = food_entity[:food_entity.index(':')]
        if '- ' in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('- ')]
            food_entity = food_entity[:food_entity.index('- ')]
        if '.' in food_entity and '(' not in food_entity:
          if food_entity.index('.') != len(food_entity) -1:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')] + food_entity[food_entity.index('.')+1:]
          else:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')]
        if len(food_entity) > 0 and food_entity[-1] == ')' and '(' not in food_entity:
            try:
                entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            except:
                print('exception')
        if '(' in food_entity:
            if ' (' in food_entity:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(' (')]
            else:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('(')]
    for food_entity in entities_list:
        if ', ' in food_entity or 'and ' in food_entity:
            for entity in re.split(', |and ',food_entity):
                    entities_list.append(entity)
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    to_remove = []
    for food_entity in entities_list:
        if food_entity[-1] == ' ':
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if food_entity[-1] == '.':
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if '\"' in food_entity or '\\"' in food_entity:
            to_remove.append(food_entity)
    entities_list = [e for e in entities_list if e not in to_remove]
    return entities_list
    
def entities_by_vicuna(answer):
    entities_list = []
    while answer[0] == '\n' or answer[0] == ' ':
        answer = answer[1:]
    if len(answer.split()) == 1 or ('\n' not in answer and ',' not in answer):
        entities_list.append(answer)
    for idx in range(len(answer)):
        try:
            if answer[idx].isnumeric():                            #limited
                if answer[idx+1] == '.' and answer[idx+2] == ' ':
                    answer = answer[:idx] + '*' + answer[idx+2:]
                if answer[idx+1].isnumeric() and answer[idx+2] == '.':
                    answer = answer[:idx] + '*' + answer[idx+3:]
            if answer[idx+1] == ' ' and answer[idx] == '-':
                answer = answer[:idx] + '*' + answer[idx+1:]
        except:
            1
    if answer.count('\n\n') == 2: 
        indices_endl = [i for i in range(len(answer)-1) if answer[i:i+2] == "\n\n"]
        if '*' in answer[indices_endl[-1]:]:
            answer = answer[indices_endl[-1]:]
        else:
            answer = answer[indices_endl[-2]:indices_endl[-1]]
    if answer.count('\n\n') >= 3: 
        indices_endl = [i for i in range(len(answer)-1) if answer[i:i+2] == "\n\n"]
        if '*' in answer[indices_endl[-1]:]:
            answer = answer[indices_endl[-1]:]
        elif '*' in answer[indices_endl[-2]:indices_endl[-1]]:
            answer = answer[indices_endl[-2]:indices_endl[-1]]
        else:
            answer = answer[indices_endl[-3]:indices_endl[-2]]
    if ',' in answer and '*' not in answer:
        try:
            shift = -1
            for char in answer[answer.index(':'):]:
                shift += 1
                if char.isalpha():
                    break
            if ', ' in answer:
                for entity in re.split(', |and ',answer[answer.index(':') + shift:]):
                    entities_list.append(entity)
            else:
                for entity in re.split(',|and ',answer[answer.index(':') + shift:]):
                    entities_list.append(entity)
        except:
            if ', ' in answer:
              for entity in re.split(', |and ',answer):
                entities_list.append(entity)
            else:
              for entity in re.split(',|and ',answer):
                entities_list.append(entity)
    if '*' in answer:
        try:
            shift = -1
            for char in answer[answer.index(':'):]:
                shift += 1
                if char.isalpha() or char == '*':
                    if char == '*':
                        shift += 1
                    break
            for entity in answer[answer.index(':') + shift:].split('*'):
                if '\n' in entity:
                  entities_list.append(entity[1:-1])
                elif '\\n' in entity:
                  entities_list.append(entity[1:-2])
                else:
                  entities_list.append(entity[1:])
        except:
            for entity in answer.split('*'):
                if '\n' in entity:
                  entities_list.append(entity[1:-1])
                elif '\\n' in entity:
                  entities_list.append(entity[1:-2])
                else:
                  entities_list.append(entity[1:])
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    for food_entity in entities_list:
        if ':' in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(':')]
            food_entity = food_entity[:food_entity.index(':')]
        if '- ' in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('- ')]
            food_entity = food_entity[:food_entity.index('- ')]
        if food_entity.count('\"') == 1 and food_entity[-1] == '\"' and '(' not in food_entity:
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if '.' in food_entity and '(' not in food_entity:
          if food_entity.index('.') != len(food_entity) -1:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')] + food_entity[food_entity.index('.')+1:]
          else:
            entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('.')]
        if len(food_entity) > 0 and food_entity[-1] == ')' and '(' not in food_entity:
            try:
                entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            except:
                print('exception')
        if '(' in food_entity:
            if ' (' in food_entity:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index(' (')]
            else:
                entities_list[entities_list.index(food_entity)] = food_entity[:food_entity.index('(')]
    for food_entity in entities_list:
        if ', ' in food_entity or 'and ' in food_entity:
            for entity in re.split(', |and ',food_entity):
                    entities_list.append(entity)
    while (1):                
        if '' in  entities_list:
            entities_list.remove('')
        else:
            break
    to_remove = []
    for food_entity in entities_list:
        if food_entity[-1] == ' ':
            entities_list[entities_list.index(food_entity)] = food_entity[:-1]
            food_entity = food_entity[:-1]
        if '\"' in food_entity or '\\"' in food_entity:
            to_remove.append(food_entity)
    entities_list = [e for e in entities_list if e not in to_remove]
    return entities_list

def make_food_list(entities_list,text):
  food_start_list = []                                                                                                    
  food_end_list = []
  if entities_list == ['no']:
    food_entities = ['O' for i in range(len(text.split()))]
  else:
    for food_entity in entities_list:  
          food_start = []
          food_end = []
          for i in range(len(text)):
            if text.startswith(food_entity, i):
              food_start.append(i)
              food_end.append(i + len(food_entity) - 1)  
            elif food_entity[-1] == 's' and text.startswith(food_entity[:-1], i):  #if plural entity, check singular
              food_start.append(i)
              food_end.append(i + len(food_entity) - 2)
          for food_st in food_start:
            while(1):
              if food_st != 0 and text[food_st - 1] != ' ':
                food_start[food_start.index(food_st)] -= 1
                food_st -= 1
              else:
                break
             
          for food_e in food_end:
            while(1):
              if (food_e != len(text) - 1) and text[food_e + 1] != ' ':
                food_end[food_end.index(food_e)] += 1
                food_e += 1
              else:
                break
          if type(food_start) == list:
                for start_item in food_start:
                  food_start_list.append(start_item)
                for end_item in food_end:
                  food_end_list.append(end_item)
          else:
                food_start_list.append(food_start)
                food_end_list.append(food_end)
  return food_start_list, food_end_list

def deduplicate_entities(df_food_ents):
    while(1):
      dataframe_changed = 0
      start_again = 0
      for index,row in df_food_ents.iterrows():
            for index_2, row_2 in df_food_ents.iterrows():
                if (index != index_2) and (row_2['start'] >= row['start']) and (row_2['end'] <= row['end']):
                    df_food_ents = df_food_ents.drop(labels = index_2, axis = 0)
                    df_food_ents = df_food_ents.reset_index(drop=True)
                    dataframe_changed = 1
                    start_again = 1
                    break
                elif (index != index_2) and (row_2['start'] >= row['start']) and (row_2['start'] <= row['end']):
                    df_food_ents.loc[index,'end'] = df_food_ents.loc[index_2,'end']
                    df_food_ents = df_food_ents.drop(labels = index_2, axis = 0)
                    df_food_ents = df_food_ents.reset_index(drop=True)
                    dataframe_changed = 1
                    start_again = 1
                    break
            if start_again:
              break
      if dataframe_changed == 0:
        break
    return df_food_ents          

def build_bio_format(df_food_ents, text):
      wrong_constructions = 0
      food_entities = []
      df_food_ents['printed'] = [0 for i in range(len(df_food_ents))]
      num_tokens = 0
      char_count = 0
      for token in text.split():
        num_tokens += 1        
        is_food = 0
        for index,row in df_food_ents.iterrows():
          start_food = row['start']
          end_food = row['end']
          if (char_count >= start_food) and (char_count + len(token) -1 <= end_food):   #is food
            is_food = 1
            first_token = 1
            if row['printed'] == 0:
              df_food_ents.loc[index,'printed'] = 1
              for food_token in text[start_food:end_food+1].split():
                if first_token == 0:
                  food_entities.append('I-FOOD')  
                else:
                  food_entities.append('B-FOOD')
                  first_token = 0
        if is_food == 0:
          food_entities.append('O')
        char_count += len(token) + 1
      return food_entities

def find_discrete_partial_hits(real_entities, predicted_entities,text,missed_entities):
  discrete_hits = 0
  partial_hits = 0
  for item in range(len(real_entities)):
    if 'B-' in real_entities[item]:
      ent_start = item
      if (ent_start == len(real_entities) - 1) or ('I-' not in real_entities[item+1]) :
        ent_end = ent_start
      else:
        if item + 1 == len(real_entities) - 1:
          ent_end = item + 1
          if real_entities[ent_start:ent_end+1] == predicted_entities[ent_start:ent_end+1]:
            discrete_hits += 1
          else:
            for num in range(ent_start,ent_end+1):
                if ('FOOD' in real_entities[num]) and ('FOOD' in predicted_entities[num]):
                    partial_hits += 1
                    break
          break
        for item_int in range(ent_start+1,len(real_entities) - 1):
          if real_entities[item_int+1] != real_entities[item_int]:
            ent_end = item_int
            break
          if item_int + 1 == len(real_entities) - 1:
            ent_end = item_int + 1
      if (real_entities[ent_start:ent_end+1] == predicted_entities[ent_start:ent_end+1]) and ((ent_end == len(real_entities) -1) or (predicted_entities[ent_end+1] != predicted_entities[ent_end])):
        discrete_hits += 1
      else:
        for num in range(ent_start,ent_end+1):
            if ('FOOD' in real_entities[num]) and ('FOOD' in predicted_entities[num]):
                partial_hits += 1
                break
  return discrete_hits, partial_hits, missed_entities

def evaluate_results(food_entities, bio_tags, hits, discrete_hits, partial_hits, all_entities, wrong_constructions, missed_entities, text):
  for item in bio_tags:
    if 'B-' in item:
      all_entities += 1
  if len(food_entities) != len(bio_tags):
    wrong_constructions += 1
  if food_entities ==  bio_tags:
    hits += 1
  discrete_partial_hits = find_discrete_partial_hits(bio_tags, food_entities, text, missed_entities)  
  discrete_hits += discrete_partial_hits[0]
  partial_hits += discrete_partial_hits[1]
  missed_entities = discrete_partial_hits[2]
  return hits, discrete_hits, partial_hits, all_entities, wrong_constructions, missed_entities
    
def evaluation_metrics(discrete_hits_prec,all_entities_prec,discrete_hits,all_entities,partial_hits,partial_hits_prec,wrong_constructions,error_sentences, model_name, to_dict = True, dictionary = {}):
    df_results = pd.DataFrame(index = ['food precision','food recall','food F1'], columns = ['score'])
    try:
      precision = discrete_hits_prec/(all_entities_prec)
    except:
      precision = 0
    try:
      recall = discrete_hits/(all_entities)
    except:
      recall = 0
    if (precision > 0) and (recall > 0):
      f1 = (2*precision*recall)/(precision + recall)
    else:
      f1 = 0
    df_results.loc['food recall','score'] = "{:.2f}".format(recall*100) + '%'
    df_results.loc['food precision','score'] = "{:.2f}".format(precision*100) + '%'
    df_results.loc['food F1','score'] = "{:.2f}".format(f1*100) + '%'    
    try:
      precision = (discrete_hits_prec+partial_hits_prec)/(all_entities_prec)
    except:
      precision = 0
    try:
      recall = (discrete_hits + partial_hits)/(all_entities)
    except:
      recall = 0
    if (precision > 0) and (recall > 0):
      f1 = (2*precision*recall)/(precision + recall)
    else:
      f1 = 0
    df_results.loc['food partial recall','score'] = "{:.2f}".format(recall*100) + '%'
    df_results.loc['food partial precision','score'] = "{:.2f}".format(precision*100) + '%'
    df_results.loc['food partial F1','score'] = "{:.2f}".format(f1*100) + '%'    
    #print(df_results)
    #print('\n')
    #print(wrong_constructions, 'entity lists wrongly made')
    #print('ommited', len(error_sentences), 'instances. Error sentences:', error_sentences)
    dct = df_results['score'].to_dict()
    arr_dct = []
    if dictionary != {}:
      arr_dct.append({'RECALL-FOOD':float(df_results.loc['food recall','score'][:-1])})
      arr_dct.append({'PRECISION-FOOD':float(df_results.loc['food precision','score'][:-1])})
      arr_dct.append({'F1-FOOD':float(df_results.loc['food F1','score'][:-1])})
      arr_dct.append({'RECALL-FOOD-PARTIAL':float(df_results.loc['food partial recall','score'][:-1])})
      arr_dct.append({'PRECISION-FOOD-PARTIAL':float(df_results.loc['food partial precision','score'][:-1])})
      arr_dct.append({'F1-FOOD-PARTIAL':float(df_results.loc['food partial F1','score'][:-1])})
      dictionary.update({"Evaluation-"+model_name:arr_dct})
    return dct,dictionary

def list_to_dict(all_entities_cur,df,tool_name,dct):
    all_entities_cur = [item for lst in all_entities_cur for item in lst]
    idx = -1
    food_area = 0
    toks = 0
    for i in range(len(df)):
     toks += len(df.loc[i,'text'].split())
    for i in range(len(df)):
      word_end = 0
      dictionary_arr = []
      dictionary_arr.append({'sentence': ' ' + df.loc[i,'text']})
      doc = ' ' + df.loc[i,'text']
      for token in df.loc[i,'text'].split():
        idx += 1
        position_in_text = word_end + doc.index(token)
        word_end = position_in_text + len(token)
        doc = (' ' + df.loc[i,'text'])[word_end:]
        if all_entities_cur[idx] == 'O' or (all_entities_cur[idx] == 'I-FOOD' and food_area == 0):
          food_area = 0
        elif all_entities_cur[idx] == 'B-FOOD':
          food_area = 1
          entity_starting = position_in_text
          if idx == len(all_entities_cur) - 1 or all_entities_cur[idx+1] != 'I-FOOD':
            new_start = position_in_text
            #while not txt[new_start].isalpha():
            #  new_start += 1
            new_end = position_in_text+len(token)
           # while not txt[new_end].isalpha():
           #   new_end -= 1
            dict_str = str(new_start) + '-' + str(new_end)
            dictionary_arr.append({dict_str:'FOOD'})
        elif all_entities_cur[idx] == 'I-FOOD' and all_entities_cur[idx-1] != 'O' and (idx == len(all_entities_cur) - 1 or all_entities_cur[idx+1] != 'I-FOOD') and food_area:
          new_start = entity_starting
          #while not txt[new_start].isalpha():
          #  new_start += 1
          new_end = position_in_text+len(token)
          #while not txt[new_end].isalpha():
          #  new_end -= 1
          dict_str = str(new_start) + '-' + str(new_end)
          dictionary_arr.append({dict_str:'FOOD'})
      dict_str = tool_name + '-' + str(i)
      dct.update({dict_str:dictionary_arr})
    return dct

def make_lists_period_split(df,num):
    list_tokens = []
    list_tags = []
    for i in range(num):
        list_tokens += df.loc[i,'text'].split()
        if 'tags' in df.columns:
            list_tags += df.loc[i,'tags']
    texts_by_period = []
    tags_by_period = []
    cur_text = ''
    cur_tags = []
    idx = -1
    sentences_num = 0
    for i in list_tokens:
        idx += 1
        cur_text += i + ' '
        if 'tags' in df.columns:
            cur_tags.append(list_tags[idx])
        if  idx == len(list_tokens) -1 or ((i == '.' or i[-1] == '.') and list_tokens[idx+1][0].isupper()):
            sentences_num += 1
            if 1:
                texts_by_period.append(cur_text[:-1])
                cur_text = ''
                if 'tags' in df.columns:
                    tags_by_period.append(cur_tags)
                    cur_tags = []
    return texts_by_period, tags_by_period
    
def texts_tags_no_duplicates(texts_by_period, tags_by_period):
    idx = -1
    tags_by_period_new = []
    for item in texts_by_period:
        idx += 1
        if item not in texts_by_period[:idx]:
            tags_by_period_new.append(tags_by_period[idx])
    texts_by_period = list(dict.fromkeys(texts_by_period))
    return texts_by_period, tags_by_period_new

def ask_LLM_get_entities_list(text, prompt, no_repetitions, repetition_LLM, food_entities_total, time_LLM):
      entities_list = []
      for i in range(no_repetitions):
          start = time.time()
          answer, no_response = ask_ollama(text, repetition_LLM[i],prompt)
          end = time.time()
          time_LLM += end-start
          answer = answer.lower()
          if answer == '' or no_response == 1:
              print('LLM error.')
              print('Splitting text by 2 and prompting LLM again...')
              start = time.time()
              answer, no_response = ask_ollama(text[:int(len(text)/2)], repetition_LLM[i],prompt)
              end = time.time()
              time_LLM += end-start
              answer = answer.lower()
              if answer == '' or no_response == 1:
                  print('Prompting with half-sized text failed.')
                  food_entities_total.append([])
                  return -1,0
              else:
                  if repetition_LLM[i] == 'orca2:7b':
                      entities_list += entities_by_orca(answer)
                  if repetition_LLM[i] == 'mistral:7b' or repetition_LLM[i] == 'openhermes:7b-v2.5':
                      entities_list += entities_by_mistral(answer)   
                  if repetition_LLM[i] == 'llama2:7b':
                      entities_list += entities_by_vicuna(answer)
              start = time.time()
              answer, no_response = ask_ollama(text[int(len(text)/2):], repetition_LLM[i],prompt)
              end = time.time()
              time_LLM += end-start
              answer = answer.lower()    
              if answer == '' or no_response == 1:
                  print('Prompting with half-sized text failed.')
                  food_entities_total.append([])
                  return -1,0
          if repetition_LLM[i] == 'orca2:7b':
              entities_list += entities_by_orca(answer)
          if repetition_LLM[i] == 'mistral:7b' or repetition_LLM[i] == 'openhermes:7b-v2.5':
              entities_list += entities_by_mistral(answer)   
          if repetition_LLM[i] == 'llama2:7b':
              entities_list += entities_by_vicuna(answer)
      entities_list = list(dict.fromkeys(entities_list))
      return entities_list, time_LLM

def ask_LLM_from_entities_list(noun_entities_list, prompt, no_repetitions, repetition_LLM, food_entities_total):
    entities_list = []
    for i in range(no_repetitions):
          for noun in noun_entities_list:
              answer, no_response = ask_ollama_syntactic(noun, repetition_LLM[i], prompt)
              answer = answer.lower()
              if answer == '' or no_response == 1:
                  print('LLM error.')
                  food_entities_total.append([])
                  return -1
              if 'mistral:7b' in repetition_LLM[i] or repetition_LLM[i] == 'llama2:7b' or repetition_LLM[i] == 'orca2:7b' or repetition_LLM[i] == 'openhermes:7b-v2.5':
                  if entities_one_item(answer):
                      entities_list.append(noun)
    entities_list = list(dict.fromkeys(entities_list))
    return entities_list

def syntactic_get_entities_list(text, nlp):

    entities_list = []
    try:
        doc = nlp(text)
    except:
        doc = nlp(text[:len(text)/2])   
    for sent in doc.sentences:
        prev = 'a'
        cur_entity = ''
        for word in sent.words:
            if word.upos == 'NOUN' or word.upos == 'PROPN':
                if prev == 'NOUN' or prev == 'PROPN':
                    cur_entity += ' ' + word.text
                elif prev == 'ADJ':
                    cur_entity += prev_word + ' ' + word.text
                else:
                    cur_entity = word.text
            else:
                if prev == 'NOUN' or prev == 'PROPN':
                    entities_list.append(cur_entity)
                    cur_entity = ''
            prev = word.upos
            prev_word = word.text
    return entities_list

def syntactic_get_entities_list_spacy(text, nlp):
    entities_list = []
    try:
        doc = nlp(text)
    except:
        doc = nlp(text[:len(text)/2])
    prev = 'a'
    cur_entity = ''
    for word in doc:
        if word.pos_ == 'NOUN' or word.pos_ == 'PROPN':
            if prev == 'NOUN' or prev == 'PROPN':
                cur_entity += ' ' + word.text
            elif prev == 'ADJ':
                cur_entity += prev_word + ' ' + word.text
            else:
                cur_entity = word.text
        else:
            if prev == 'NOUN' or prev == 'PROPN':
                entities_list.append(cur_entity)
                cur_entity = ''
        prev = word.pos_
        prev_word = word.text
    return entities_list

def find_entities_in_text(entities_list,predicted_entities_total, food_entities_total,text,print_df):
      for item in entities_list:
          if item not in predicted_entities_total:
              predicted_entities_total.append(item)
      food_start_list, food_end_list = make_food_list(entities_list,text)
      df_food_ents = pd.DataFrame(columns = ['start','end'])    
      df_food_ents['start'] = food_start_list
      df_food_ents['end'] = food_end_list
      df_food_ents = deduplicate_entities(df_food_ents)
      if print_df:
          print(df_food_ents)         
      food_entities = build_bio_format(df_food_ents, text)
      food_entities_total.append(food_entities)
      return food_entities

def update_df_scores(df_scores, hits, discrete_hits, discrete_hits_prec, partial_hits, partial_hits_prec, wrong_constructions, all_entities, all_entities_prec, error_sentences, food_entities_total, predicted_entities_total, missed_entities, fp_entities,set_foods, set_no_foods):
    df_scores.loc['hits','value'] = hits
    df_scores.loc['discrete_hits','value'] = discrete_hits
    df_scores.loc['discrete_hits_prec','value'] = discrete_hits_prec
    df_scores.loc['partial_hits','value'] = partial_hits 
    df_scores.loc['partial_hits_prec','value'] = partial_hits_prec
    df_scores.loc['wrong_constructions','value'] = wrong_constructions
    df_scores.loc['all_entities','value'] = all_entities
    df_scores.loc['all_entities_prec','value'] = all_entities_prec 
    df_scores.loc['error_sentences','value'] = error_sentences
    df_scores.loc['food_entities_total','value'] = food_entities_total
    df_scores.loc['predicted_entities_total','value'] = predicted_entities_total
    df_scores.loc['missed_entities','value'] = missed_entities
    df_scores.loc['fp_entities','value'] = fp_entities
    df_scores.loc['set_foods','value'] = set_foods
    df_scores.loc['set_no_foods','value'] = set_no_foods
    return df_scores

def get_info_from_df(df_scores):
    hits = df_scores.loc['hits','value']
    discrete_hits = df_scores.loc['discrete_hits','value']
    discrete_hits_prec = df_scores.loc['discrete_hits_prec','value']
    partial_hits = df_scores.loc['partial_hits','value']
    partial_hits_prec = df_scores.loc['partial_hits_prec','value']
    wrong_constructions = df_scores.loc['wrong_constructions','value']
    all_entities = df_scores.loc['all_entities','value']
    all_entities_prec = df_scores.loc['all_entities_prec','value']
    error_sentences = df_scores.loc['error_sentences','value']
    food_entities_total = df_scores.loc['food_entities_total','value']
    predicted_entities_total = df_scores.loc['predicted_entities_total','value']
    missed_entities = df_scores.loc['missed_entities','value']
    fp_entities = df_scores.loc['fp_entities','value']
    start = df_scores.loc['start_time','value']
    set_foods = df_scores.loc['set_foods','value']
    set_no_foods =  df_scores.loc['set_no_foods','value']
    return hits, discrete_hits, discrete_hits_prec, partial_hits, partial_hits_prec, wrong_constructions, all_entities, all_entities_prec, error_sentences, food_entities_total, predicted_entities_total, missed_entities, fp_entities, start, set_foods, set_no_foods

def df_scores_to_dict(df_scores,dictionary,model_name):
    hits, discrete_hits, discrete_hits_prec, partial_hits, partial_hits_prec, wrong_constructions, all_entities, all_entities_prec, error_sentences, food_entities_total, predicted_entities_total, missed_entities, fp_entities, start, set_foods, set_no_foods = get_info_from_df(df_scores)
    dict_results,dictionary = evaluation_metrics(discrete_hits_prec,all_entities_prec,discrete_hits,all_entities,partial_hits,partial_hits_prec,wrong_constructions,error_sentences, model_name = model_name,dictionary = dictionary, to_dict = True)
    return dict_results,dictionary
    
def LLM_foodNER(df, df_scores, texts, true_tags, end_text, no_repetitions, repetition_LLM, prompt = False, nlp = False, syntactic = False, start_text = 0, pos_tagger = 'stanza'):
    texts_missed = []
    missed_entities_old = []
    i = -1
    hits, discrete_hits, discrete_hits_prec, partial_hits, partial_hits_prec, wrong_constructions, all_entities, all_entities_prec, error_sentences, food_entities_total, predicted_entities_total, missed_entities, fp_entities, start, set_foods, set_no_foods = get_info_from_df(df_scores)
    llm_prompts = 0
    llm_prompts_list = []
    end = start
    time_LLM = 0
    for text in texts:
      i += 1
      if i <= start_text -1:
            continue
      text = text.lower() #!
      if not syntactic:
          entities_list, time_LLM = ask_LLM_get_entities_list(text, prompt, no_repetitions, repetition_LLM, food_entities_total, time_LLM)
          if time_LLM == 0:
              error_sentences.append(i)
      else:
          if not nlp:
              print('Error: nlp argument missing.')
              return pd.DataFrame()
          if pos_tagger == 'spacy':
              noun_entities_list = syntactic_get_entities_list_spacy(text,nlp)
          elif pos_tagger == 'stanza':
              noun_entities_list = syntactic_get_entities_list(text, nlp)
          else:
              print('Error: pos tagger not supported. Available taggers: spacy,stanza')
              return pd.DataFrame()
          new_noun_entities_list = []
          entities_list = []
          for item in noun_entities_list:
              if item not in set_foods and item not in set_no_foods:
                  if nlp(item).ents == []:
                      new_noun_entities_list.append(item)
                  else:
                      found = 0
                      for ent in nlp(item).ents:
                          if ent.text == item:
                              found = 1
                          if found == 0:
                              new_noun_entities_list.append(item)
              elif item in set_foods:
                  entities_list.append(item)
          if new_noun_entities_list != []:
              llm_prompts += len(new_noun_entities_list)
              entities_list += ask_LLM_from_entities_list(new_noun_entities_list, prompt, no_repetitions, repetition_LLM, food_entities_total) 
              for item in entities_list:
                  if item not in set_foods:
                      set_foods.add(item)
              for item in new_noun_entities_list:
                  if item not in entities_list:
                      set_no_foods.add(item)
      if entities_list == -1:
          if i >= end_text:
              break
          continue
      if 'no' in entities_list:
          entities_list.remove('no')
      if i % 2 == 0:
          llm_prompts_list.append(llm_prompts)
          llm_prompts = 0
      food_entities = find_entities_in_text(entities_list,predicted_entities_total,food_entities_total, text, print_df = False)
      if 'tags' in df.columns:
        missed_entities_old+= missed_entities
        hits, discrete_hits, partial_hits, all_entities, wrong_constructions, missed_entities = evaluate_results(food_entities, true_tags[i], hits, discrete_hits, partial_hits, all_entities, wrong_constructions,missed_entities, text)
        if (len(missed_entities) - len(missed_entities_old)) >= 1:
            texts_missed.append(len(text)) 
        hits_trash, discrete_hits_prec, partial_hits_prec, all_entities_prec, wrong_constructions_trash, fp_entities = evaluate_results(true_tags[i], food_entities, hits, discrete_hits_prec, partial_hits_prec, all_entities_prec, wrong_constructions, fp_entities, text)
      if i >= end_text:
        break
    end = time.time()
    print('time elapsed: {0:.2f}'.format(end-start), 'seconds')
    print('time consumed by LLM: {0:.2f}'.format(time_LLM), 'seconds')
    df_scores = update_df_scores(df_scores, hits, discrete_hits, discrete_hits_prec, partial_hits, partial_hits_prec, wrong_constructions, all_entities, all_entities_prec, error_sentences, food_entities_total, predicted_entities_total, missed_entities, fp_entities, set_foods, set_no_foods)
    plot_llm_prompts_by_time(llm_prompts_list)
    print('texts missed:', texts_missed)
    return df_scores

def make_df_fp(fp_entities):
    df_fp = pd.DataFrame()
    df_fp['entity'] = list(dict.fromkeys(fp_entities))
    df_fp['# fp'] = [fp_entities.count(a) for a in list(dict.fromkeys(fp_entities))]
    return df_fp
    
def make_df_misses(missed_entities):
    df_misses = pd.DataFrame()
    df_misses['entity'] = list(dict.fromkeys(missed_entities))
    df_misses['# fn'] = [missed_entities.count(a) for a in list(dict.fromkeys(missed_entities))]
    df_misses['# fn'].sum()
    df_misses.sort_values(by='# fn',ascending=False)[:20]
    return df_misses

def save_results(file_name, food_entities_total):
    name = 'results/' + file_name
    with open(name,'w+') as f:
    	for i in food_entities_total:
    		f.write('%s\n'%i)

def make_df_scores(set_foods,set_no_foods):
    df_scores = pd.DataFrame(index = ['hits', 'discrete_hits', 'discrete_hits_prec','partial_hits','partial_hits_prec', 'wrong_constructions', 'all_entities', 'all_entities_prec', 'error_sentences', 'food_entities_total', 'predicted_entities_total', 'missed_entities','fp_entities', 'start_time', 'set_foods', 'set_no_foods'], columns = ['value'])
    for i in range(len(df_scores)):
        if len(df_scores[:'error_sentences']) - 1 > i:
            df_scores.iloc[i,0] = 0
        else:
            df_scores.iloc[i,0] = []
    df_scores.loc['set_foods','value'] = set_foods
    df_scores.loc['set_no_foods','value'] = set_no_foods
    df_scores.loc['start_time','value'] = time.time()
    return df_scores

def update_time(df):
    df.loc['start_time','value'] = time.time()
    return df
    
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def annotate_text(list_texts,how_many,predicted_entities):
    i = -1
    for text in list_texts[:how_many]:
        i += 1
        text = text.lower()
        food_start_list, food_end_list = make_food_list(predicted_entities,text)
        df_food_ents = pd.DataFrame(columns = ['start','end'])    
        df_food_ents['start'] = food_start_list
        df_food_ents['end'] = food_end_list
        df_food_ents = deduplicate_entities(df_food_ents)
        food_start_list = df_food_ents['start']
        food_end_list = df_food_ents['end']
        food_start_list = sorted(food_start_list, reverse=False) 
        food_end_list = sorted(food_end_list, reverse=False) 
        text_end = 0
        for idx in range(len(food_start_list)):
            print(text[text_end:food_start_list[idx]], f"{bcolors.OKGREEN}%s{bcolors.ENDC}"% text[food_start_list[idx]:food_end_list[idx]+1],end = '')
            text_end = food_end_list[idx]+1
        print(text[text_end:])
        print('\n')

def plot_llm_prompts_by_time(llm_prompts_list):
    x = [i*2 for i in range(1,len(llm_prompts_list)+1)]
    plt.title("LLM prompts per texts seen", fontsize='16')
    plt.plot(x, llm_prompts_list)	
    plt.xlabel("# texts seen",fontsize='13')	
    plt.ylabel("LLM prompts",fontsize='13')	
    plt.legend(('YvsX'),loc='best')	
    plt.grid()	
    plt.show()

def save_set(set_foods, file_path):
    with open(file_path,'w') as f:
        for item in set_foods:
            f.write(str(item))

def read_set(file):
    with open(file, "r") as txt_file:
          a = txt_file.readlines()
          lst = a[0][1:a[0].index("}value")][1:-1].split('\', \'')
          set = {item for item in lst}
    return set

def clean_text(texts_by_period):
    idx = -1
    for text in texts_by_period:
        idx += 1
        texts_by_period[idx] = texts_by_period[idx].replace('\"','’')
        texts_by_period[idx] = texts_by_period[idx].replace('\'','’')
        #texts_by_period[idx] = texts_by_period[idx].replace('>','')
        #texts_by_period[idx] = texts_by_period[idx].replace('<','')
        #texts_by_period[idx] = texts_by_period[idx].replace('/','\/')
    return texts_by_period

def food_data_to_csv(df,all_entities_cur,tool,df_to_merge,discard_non_entities = False):
  output_df = pd.DataFrame()
  list_ids = []
  list_text_ids = []
  list_phrases = []
  list_tags = []
  list_positions = []
  list_products = []
  i =  0
  for idx in range(len(all_entities_cur)):
    if all_entities_cur[idx] == 'I-FOOD' and (all_entities_cur[idx-1] == 'O' or (all_entities_cur[idx-1] == 'I-FOOD' and all_entities_cur[idx-2] == 'O')):
      i+= 1
  idx = 0
  for text in df['text']:
    for token in text.split():
      idx += 1
  cur_phrase = ''
  idx = -1  
  idx_text = -1
  idx_phrase = -1
  
  for text in df['text']:
    word_end = -1
    doc = text
    idx_text += 1
    for token in text.split():
      idx += 1
      position_in_text = word_end + doc.index(token)
      word_end = position_in_text + len(token)
      doc = text[word_end:]
      if all_entities_cur[idx] == 'O' or (all_entities_cur[idx] == 'I-FOOD' and food_area == 0):
       if not discard_non_entities:
        idx_phrase += 1
        list_phrases.append(token)
        list_text_ids.append(idx_text)
        list_products.append(None)
        list_ids.append(tool + '-' + str(idx_phrase))
        list_tags.append('O')
        list_positions.append(str(position_in_text) + '-' + str(position_in_text+len(token)-1))
       food_area = 0
      elif all_entities_cur[idx] == 'B-FOOD':
        cur_phrase = token
        idx_phrase += 1
        food_area = 1
        entity_starting = position_in_text
        if idx == len(all_entities_cur) - 1 or all_entities_cur[idx+1] != 'I-FOOD':
          list_phrases.append(cur_phrase)
          list_text_ids.append(idx_text)
          if 'product' in list(df.columns):
           list_products.append(df.loc[idx_text,'product'])
          else:
           list_products.append(None)
          list_ids.append(tool + '-' + str(idx_phrase))
          list_tags.append('FOOD')
          list_positions.append(str(position_in_text) + '-' + str(position_in_text+len(token)-1))
      elif all_entities_cur[idx] == 'I-FOOD' and all_entities_cur[idx-1] != 'O':
        cur_phrase += ' ' + token
        if (idx == len(all_entities_cur) - 1 or all_entities_cur[idx+1] != 'I-FOOD') and food_area:
          list_phrases.append(cur_phrase)
          list_text_ids.append(idx_text)
          if 'product' in list(df.columns):
           list_products.append(df.loc[idx_text,'product'])
          else:
           list_products.append(None)
          list_ids.append(tool + '-' + str(idx_phrase))
          list_tags.append('FOOD')
          list_positions.append(str(entity_starting) + '-' + str(position_in_text+len(token)-1))
  output_df['phrase_id'] = list_ids
  output_df['text_id'] = list_text_ids
  output_df['phrase'] = list_phrases
  output_df['tag'] = list_tags
  output_df['position'] = list_positions
  output_df['ground truth'] = [None for i in range(len(list_phrases))]
  output_df['food product'] = list_products
  if not df_to_merge.empty:
    output_df = pd.concat([df_to_merge,output_df],ignore_index = True)
  return output_df

def generic_data_to_csv(df,output_df,food_df,discard_non_entities = False):
  new_df = pd.DataFrame()
  list_ids = []
  list_text_ids = []
  list_phrases = []
  list_tags = []
  list_positions = []
  cur_phrase = ''
  for tool in list(output_df.columns):
      cur_phrase = ''
      idx = -1
      idx_text = -1
      idx_phrase = -1
      all_entities_cur = [tag for arr in output_df.loc[:,tool] for tag in arr]
      for text in df['sentence_non_tokenized']:
        idx_text += 1
        doc = text[1:]
        word_end = 0
        for token in text.split():
          idx += 1
          position_in_text = word_end + doc.index(token)
          word_end = position_in_text + len(token)
          doc = text[1:][word_end:]
          if all_entities_cur[idx] == 'O':
           if not discard_non_entities:
            idx_phrase += 1
            list_phrases.append(token)
            list_text_ids.append(idx_text)
            list_ids.append(tool + '-' + str(idx_phrase))
            list_tags.append('O')
            list_positions.append(str(position_in_text) + '-' + str(position_in_text+len(token)-1))
           food_area = 0
          elif 'B-' in all_entities_cur[idx]:
            cur_phrase = token
            idx_phrase += 1
            food_area = 1
            entity_starting = position_in_text
            if idx == len(all_entities_cur) -1 or 'I-' not in all_entities_cur[idx+1]:
              list_phrases.append(cur_phrase)
              list_text_ids.append(idx_text)
              list_ids.append(tool + '-' + str(idx_phrase))
              list_tags.append(all_entities_cur[idx][2:])
              list_positions.append(str(position_in_text) + '-' + str(position_in_text+len(token)-1))
          elif 'I-' in all_entities_cur[idx] and 'O' not in all_entities_cur[idx-1]:
            cur_phrase += ' ' + token
            if (idx == len(all_entities_cur) -1 or 'I-' not in all_entities_cur[idx+1]) and food_area:
              list_phrases.append(cur_phrase)
              list_text_ids.append(idx_text)
              list_ids.append(tool + '-' + str(idx_phrase))
              list_tags.append(all_entities_cur[idx][2:])
              list_positions.append(str(entity_starting) + '-' + str(position_in_text+len(token)-1))
  new_df['phrase_id'] = list_ids
  new_df['text_id'] = list_text_ids
  new_df['phrase'] = list_phrases
  new_df['tag'] = list_tags
  new_df['position'] = list_positions
  new_df['ground truth'] = [None for i in range(len(list_phrases))]
  new_df['food product'] = [None for i in range(len(list_phrases))]
  if not food_df.empty:
    new_df = pd.concat([new_df,food_df],ignore_index = True)
  return new_df

def annotate_text_tokens(df,all_entities_merged,N):

  lst = []
  all_entities_cur = all_entities_merged
  id = -1
  for text in df['text'][:N]:
        id += 1
        lst.append(all_entities_cur[:len(text.split())])
        all_entities_cur = all_entities_cur[len(text.split()):]
  return lst

def read_configuration_file(conf_path):
  config = ConfigParser()
  config.read(conf_path)
  config_data = config['parameters']
  dataset = config_data['dataset']
  text_column = config_data['text_column']
  ground_truth_column = config_data['ground_truth_column']
  product_column = config_data['product_column']
  csv_delimiter = config_data['csv_delimiter']
  prediction_values = eval(config_data['prediction_values'])
  N = int(config_data['N'])
  minio = config_data['minio']
  return dataset, text_column, ground_truth_column, product_column, csv_delimiter, prediction_values, N, minio

def read_configuration_file_ont(conf_path):
  config = ConfigParser()
  config.read(conf_path)
  config_data = config['parameters']
  try:
   dataset = config_data['dataset']
   ontology_file = config_data['ontology_file']
   ontology_header = int(config_data['ontology_header'])
   ontology_col_id = config_data['ontology_col_id']
   ontology_col_text = config_data['ontology_col_text']
   ontology_col_separator = config_data['ontology_col_separator']
   ontology_text_separator = config_data['ontology_text_separator']
   delta_alg = int(config_data['delta_alg'])
   similarity = config_data['similarity']
  except:
   return [None for i in range(8)]
  return ontology_file, ontology_header, ontology_col_id, ontology_col_text, ontology_col_separator, ontology_text_separator, delta_alg, similarity

def generate_output_file_name(dataset,prediction_values):
 keys = list(prediction_values.keys())
 values = list(prediction_values.values())
 values = [item for sublist in values for item in sublist]
 str_keys,str_values = '',''
 for item in keys:
  str_keys += item + '_'
 for item in values:
  str_values += item + '_'
 str_keys,str_values = str_keys[:-1],str_values[:-1]
 if (not os.path.exists("results/")):
  os.mkdir('results/')
 name = 'results/' + dataset + '_' + str_keys + '_' + str_values
 return name
