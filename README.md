# Overview

Software that performs [named entity recognition (NER)] and optionally [named entity linking (NEL)] on input texts and visualizes results. It consists of two separate components, one for named entity recognition and one for visualization of results (GUI). Entities and models supported are the following:

| | spaCy | spaCy + RoBERTa | Flair | Stanza | InstaFoodRoBERTa | SciFoodNER | LLMs: Mistral-7B, Llama2-7b, Openhermes:7b-v2.5
| ----------- |  ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- 
| FOOD| |  |  | | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:
| PERSON | :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | | | 
| NORP   | :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| FAC | :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| PRODUCT| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| ORG | :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| GPE | :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| LOC | :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| DATE| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| TIME| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| PERCENT| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| CARDINAL| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| MONEY| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| QUANTITY| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| ORDINAL| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| EVENT| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| WORK_OF_ART| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| LAW| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| LANGUAGE| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 
| MISC| :heavy_check_mark:| :heavy_check_mark:  | :heavy_check_mark: |:heavy_check_mark: | | | 


## Named entity recognition (NER)

This component's input is a CSV file containing texts we want to perform NER on, optionally containing ground truth data. Its output is a CSV and a JSON file containing NER results.

### Parameters

1. _dataset_ (str): path to input CSV file 
2. _prediction_values_ (dict): models-entity type pairs: {"cardinal":['flair','spacy_roberta'],"food":['instafoodroberta']} denotes Flair and SpaCy-RoBERTa for cardinal entities, InstaFoodRoBERTa-ner for food entities
3. _text\_column_ (str): name of _dataset_ column containing texts
4. _ground\_truth\_column_ (optional,str):  name of _dataset_ column containing ground truth tags
5. _product\_column_ (optional,str):  name of _dataset_ column containing product that describes text
6. _csv\_delimiter_ (str): input CSV file delimiter
7.  _N_ (int/str): number of texts to be annotated or 'all' for the whole dataset
8. _syntactic\_analysis\_tool_ (optional,str): name of model to extract nouns/noun phrases from text. Can be _spacy_ or _stanza_
9. _prompt\_id_ (optional, int): prompt id (_0,1,2 or 3_). Used in the case of food entity extraction using an LLM
10. _minio_ (optional,str): credentials for minio server. Used when dataset is not local, but instead is an s3 path to the minio server

The input parameters can be adjusted in the _config\_file.ini_ configuration file.

### Entity linking parameters

If you want to perform entity linking to an input ontology, you should add the following parameters in _config\_file.ini_ configuration file:

11. _ontology\_file_ (optional,str): path to ontology CSV file
12. _ontology\_header_ (optional,int): row number of _ontology\_file_ containing column labels (e.g. -1)
13. _ontology\_col\_id_ (optional,str): _ontology\_file_ column containing ontology item ids
14. _ontology\_col\_text_ (optional,str):  _ontology\_file_ column containing ontology item names
15. _ontology\_col\_separator_ (optional,str):  _ontology\_file_ delimiter
16. _ontology\_text\_separator_ (optional,str): token separator of an ontology item (e.g. ' ') 
17. _delta\_alg_ (optional,int): threshold for [PyTokenJoin], e.g. 1
18. _similarity_ (optional,str): similarity function for [PyTokenJoin], _jaccard_ or _edit_

### Functionality

After the input file is processed, entity recognition models are loaded. Those models predict named entity labels (locations, organizations etc.) and/or food entity labels and insert the results in a Python dictionary. Currently, the software supports four generic NER models: [spaCy], [spaCy + RoBERTa benchmark], [Flair] and [Stanza (StanfordNLP)] and the following food NER models: [SciFoodNER], [InstaFoodRoBERTa] and LLMs. It can be extended to use more models in the future. In cases where ground truth is available within the input file, evaluation of results is performed for all models and metrics are also reported in the resulting dictionary (precision, recall, F1 score). If entity linking parameters are given, entity linking is performed using [PyTokenJoin]. The dictionary is then written on an output JSON file, as well as an output CSV file. The two files are stored under the name the user provided, with the extensions .json and .csv, respectively. 

#### LLMs

The chosen LLM or set of LLMs is prompted to identify food entities in every text given, with one of the following prompts (_prompt\_id_):
1. _'Print only one comma-separated list of the foods, drinks or edible ingredients mentioned in the previous text. Do write a very short answer, with no details, just the list. If there are no foods, drinks or edible ingredients mentioned, print no.'_
2. _'You are a food allergy specialist and your task is to find anything edible, i.e. food, drink or ingredient, mentioned in the previous text. If you lose any edible item mentioned, there is a risk of someone getting allergy and you will be penalized. Print the edible items you found in a comma-separated list, each edible item printed separately and without further information. If there are no edible items mentioned in the text, print no.'_
3. _'Find any foods, drinks or edible ingredients mentioned in the previous text. Print them in a comma-separated list. If there are none, print no. Write a short answer.'_

You can set the prompt id (1,2 or 3) by adjusting the _prompt\_id_ parameter. 
In the case of many LLMs, the union of answers is considered as the set of food entities returned.

You can also use _spaCy_ or _Stanza_ (set it in argument _syntactic\_analysis\_tool_) as a noun/noun phrase extractor by setting _prompt\_id_ to 0. In that case, the LLM(s) will <u>instead</u> be prompted to classify the extracted nouns/noun phrases are food entities or non-food entities, with the following prompt:

_'Classify the following item as EDIBLE or NON EDIBLE. Desired format: [EDIBLE/NON EDIBLE]. Input:'_

## Visualization

There are 2 GUIs available. The first one is for viewing results from the first component, filter out entities and view statistics. The second one is also for viewing results from the first component and adding any missed entities (manual correction). The first GUI is described in [src/frontend/README.md], while the second GUI is described in [src/frontend_extraction_deduplication/README.md].

# Installation

## Prerequisites

You can run this software on any machine that has Python 3 and pip installed.

##
Download this repository by running:

```sh
$ git clone https://github.com/stelar-eu/GenericNER.git
```

Then, move to directory:
```sh
$ cd GenericNER
```

First, you need to download all environments and packages stated on [requirements.txt] file. To do that, run:

```sh
$ pip install -r requirements.txt
```
Additionally, you will need to download the following spaCy models:

```sh
$ python3 -m spacy download en_core_web_trf
$ python3 -m spacy download en_core_web_sm
```

To perform food NER using SciFoodNER, you will need to download _cafeteria_ directory from [here] and into _src_ directory. 

To perform food NER using LLMs, you will need to have [Ollama] installed and _mistral:7b_, _llama2:7b_, _openhermes:7b-v2.5_ models pulled. Directions can be found on [Ollama's GitHub page]. 

# Execution

You are now ready to run the application. 

- Step 1 - Named entity recognition (NER)

In _src_ directory, run: 
```sh
$ python3 main.py
```

<!---
where _your_csv_ is the input CSV file. If you have no CSV file available and you want to try the application, you can give 'default' in place of _your_csv_ and the application will run on a sample of conll2012_ontonotesv5_ dataset from [HuggingFace].


Expect this step to take a few minutes to complete the first time, since models need to be downloaded and imported.

A JSON file is now generated in the output path given (_output\_file_).  The JSON file generated by the software when run with the _default_ parameter is the file [output.json.example]. The output path can be changed to what the user prefers, but the path must be written into single quotation marks (''). This file contains the data from the entity extraction, i.e. the named entities and their labels as identified by each tool, as well as evaluation metrics, if ground truth was provided. 
-->

- Step 2 - Visualization

Follow the README instructions of the respective GUI you want to run.

This software was developed under [STELAR] project.

   [spaCy]: <http://spacy.io>
   [spaCy + RoBERTa benchmark]: <https://spacy.io/models/en#en_core_web_trf>
   [Stanza (StanfordNLP)]: <https://nlp.stanford.edu/software/>
   [Flair]: <https://github.com/flairNLP/flair/>
   [STELAR]: <http://stelar-project.eu>
   [Anaconda]: <https://www.anaconda.com/>
   [Streamlit]: <https://streamlit.io/>
   [HuggingFace]: <https://huggingface.co/datasets/conll2012_ontonotesv5>
   [BIO format]: <https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)>
   [configuration file]: <config_file.ini>
   [output.json.example]: <examples/output.json.example>
   [guide.pdf]: <docs/guide.pdf>
   [here]: <https://portal.ijs.si/nextcloud/s/C3jCDq84TBoE8gY>
   [sample_groundtruth.csv]: <examples/sample_groundtruth.csv>
   [sample_no_groundtruth.csv]: <examples/sample_no_groundtruth.csv>
   [requirements.txt]: <requirements.txt> 
   [named entity recognition (NER)]: <https://en.wikipedia.org/wiki/Named-entity_recognition>
   [src/frontend_extraction_deduplication/README.md]: <src/frontend_extraction_deduplication/README.md>
   [src/frontend/README.md]: <src/frontend/README.md>
   [named entity linking (NEL)]: <https://en.wikipedia.org/wiki/Entity_linking#:~:text=In%20natural%20language%20processing%2C%20entity,as%20famous%20individuals%2C%20locations%2C%20or>
   [SciFoodNER]: <https://github.com/gjorgjinac/SciFoodNER/>
   [InstaFoodRoBERTa]: <https://huggingface.co/Dizex/InstaFoodRoBERTa-NER>
   [Ollama]: <https://ollama.com/>
   [Ollama's GitHub page]: <https://github.com/ollama/ollama>
   [PyTokenJoin]: <https://github.com/alexZeakis/pyTokenJoin>
