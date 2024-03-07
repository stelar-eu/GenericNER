# Overview

Software that performs [named entity recognition (NER)] on input texts and visualizes results. It consists of two separate components, one for the named entity recognition and one for visualization of results (GUI). Entities recognized are the following:

| Available entity types | | | | |
| ----------- |  ----------- | ----------- | ----------- | ----------- |
| PERSON     | ORG  |  TIME | ORDINAL |LAW
| NORP   |GPE   |  PERCENT | CARDINAL | LANGUAGE
| FAC |  LOC   |  MONEY | EVENT | MISC
| PRODUCT| DATE | QUANTITY | WORK_OF_ART | FOOD

## Named entity recognition (NER)

This component's input is a CSV file containing texts we want to perform NER on, optionally containing ground truth data. Its output is a CSV file containing NER results.

### Input 

1. _extraction\_type_: 'generic','food' or 'all'
2. _dataset_: path to input CSV file
3. _model_: 'scifoodner' or 'instafoodroberta' or an array of LLMs,e.g. ['mistral_7b'], ['llama2:7b','openhermes:7b-v2.5']
4. _minio_
5. _text\_column_: name of column containing texts
6. _ground\_truth\_column (optional)_:  name of column containing ground truth tags
7. _csv\_delimiter_: input CSV delimiter
8. _output\_file_: name of output file
9. _N_: number of texts to be annotated 

### Functionality

After the input file is processed, entity recognition models are loaded. Those models predict named entity labels (locations, organizations etc.) and insert the results in a Python dictionary. Currently, the software supports four NER models: [spaCy], [spaCy + RoBERTa benchmark], [Flair] and [Stanza (StanfordNLP)], but it can be extended to using more models in the future. In case ground truth is available within the input file, evaluation of results is performed for all models and metrics are also reported in the resulting dictionary. The dictionary is then written on an output JSON file.

## Visualization

In order to visualize and compare entity labels identified by various NER models in the NER component, we developed a web-based graphical user interface (GUI). 

Implemented in Python using the [Streamlit] package, this GUI allows the user to interactively inspect the entities identified by each model. The user can choose to print a text based on multiple filters (e.g., which sentences contain LOC entities) or view annotations only for specific labels of their choice (e.g., only locations or organizations or only annotations by spaCy). The user can also display various statistics, i.e., how many different entities were detected by each tool as well as evaluation metrics, if ground truth is available.

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
$ cd GenericNER-main
```

First, you need to download all environments and packages stated on [requirements.txt] file. To do that, run:

```sh
$ pip install -r requirements.txt
```
Additionally, you will need to download the following spaCy models:

```sh
$ python -m spacy download en_core_web_trf
$ python -m spacy download en_core_web_sm
```

# Execution

You are now ready to run the application. 

- Step 1 - Named entity recognition (NER)

In _src_ directory, run: 
```sh
$ python3 entity_extraction.py
```

<!---
where _your_csv_ is the input CSV file. If you have no CSV file available and you want to try the application, you can give 'default' in place of _your_csv_ and the application will run on a sample of conll2012_ontonotesv5_ dataset from [HuggingFace].
-->

Expect this step to take a few minutes to complete the first time, since models need to be downloaded and imported.

A JSON file is now generated in the output path given (_output\_file_).  The JSON file generated by the software when run with the _default_ parameter is the file [output.json.example]. The output path can be changed to what the user prefers, but the path must be written into single quotation marks (''). This file contains the data from the entity extraction, i.e. the named entities and their labels as identified by each tool, as well as evaluation metrics, if ground truth was provided. 

- Step 2 - Visualization

For the GUI, in _frontend_ directory run:

```sh
$ streamlit run entity_extraction_readonly.py
```

The application will open on a new tab on your web browser and will look like this:

![alt text](https://github.com/VasiPitsilou/NLP/blob/2cac91cfa9f69499a82797614cd78fdec5229763/image.png?raw=true)

Upload the JSON file generated in the first step and browse the application. A small guide for its usage can be found here: [guide.pdf]. 

This software was developed under the [STELAR] project.

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
   [sample_groundtruth.csv]: <examples/sample_groundtruth.csv>
   [sample_no_groundtruth.csv]: <examples/sample_no_groundtruth.csv>
   [requirements.txt]: <requirements.txt> 
   [named entity recognition (NER)]: <https://en.wikipedia.org/wiki/Named-entity_recognition>
