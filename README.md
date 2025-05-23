# Welcome

To the NER GUI developed for EU-funded STELAR project.

## --README file under construction--

## Functionality

This library performs translation, summarization, entity recognition (NER), main entity selection and entity linking (EL) on a given text.

### Translation

Translation of text in English. Two methods available:
1. deep-translator library
2. LLM

### Summarization

Summarization of given text. Method available:
1. LLM

### Named Entity Recognition (NER)

NER on given text for types specified by user. Available methods:
1. InstaFoodRoBERTa (for food entities only)
2. LLM

### Main Entity Selection (MES)

Selection of single or multiple main entities that the text refers to. Available methods:
1. LLM

### Entity Linking

Linking of entities returned by NER module to k closest ontology entities. Ontology is given by the user. Available methods:
1. ChromaDB
2. bm25s
3. LLM-augmented ChromaDB
4. LLM-augmented bm25s
5. LLM

## Usage

