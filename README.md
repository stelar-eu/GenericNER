# Welcome

To the Named Entity Recognition and Linking pipeline UI.
![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](Figures/dashboard.png)


## Functionality

This library performs a pipeline of translation, summarization, entity recognition (NER), main entity selection and entity linking (EL) on a given text.

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](Figures/pipeline.png)


### 1. Translation

Translation of text in English. Two methods available:
* [deep-translator](https://github.com/nidhaloff/deep-translator) library
*  LLM
    * Locally ([Ollama](https://github.com/ollama/ollama))
    * Via API ([Groq](groq.com))  

  <ins>Note 1</ins>: If you want to run an Ollama model, you must have a running Ollama docker container. <br>
  <ins>Note 2</ins>: If you want to run a Groq model, please enter your Groq API key in _scr/.env_ file and _src/archive/.env_ file.

### 2. Summarization

Summarization of given text. Method available:
*  LLM

### 3. Named Entity Recognition (NER)

NER on given text for types specified by user. Methods available:
* InstaFoodRoBERTa (for food entities only)
* LLM

### 4. Main Entity Selection (MES)

Selection of single or multiple main entities that the text refers to. Method available:
* LLM

### 5. Entity Linking

Linking of entities returned by NER module to k closest ontology entities. Ontology is given by the user. Methods available:
* [ChromaDB](https://www.trychroma.com/)
* [bm25s](https://bm25s.github.io/)
* LLM-augmented ChromaDB
* LLM-augmented bm25s
* LLM 

# Usage

### Installation

To install, clone the current repository. Navigate to GenericNER-main:

```
cd GenericNER-main
```

It is recommended to activate a new Python 3.12 virtual environment and activate it. 

```
python -m venv $VENV_PATH
source activate $VENV_PATH/bin/activate
```

To deactivate it:
```
deactivate
```

Install requirements:

```
pip install -r requirements.txt
```

Run dashboard:

```
streamlit run src/Dashboard.py
```

You're all set!

# Credits

Developed for EU-funded [STELAR project](https://stelar-project.eu/).
