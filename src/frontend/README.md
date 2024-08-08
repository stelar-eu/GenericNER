In order to visualize and compare entity labels identified by various NER models in the NER component, we developed a web-based graphical user interface (GUI). 

Implemented in Python using [Streamlit] package, this GUI allows the user to interactively inspect the entities identified by each model. The user can choose to print a text based on multiple filters (e.g., which sentences contain LOC entities) or view annotations only for specific labels of their choice (e.g., only locations or organizations or only annotations by spaCy). The user can also display various statistics, i.e., how many different entities were detected by each tool as well as evaluation metrics, if ground truth is available.

# Usage

In _frontend_ directory run:

```sh
$ streamlit run entity_extraction_readonly.py
```

The application will open on a new tab on your web browser and will look like this:

![alt text](https://github.com/VasiPitsilou/NLP/blob/2cac91cfa9f69499a82797614cd78fdec5229763/image.png?raw=true)

Upload the JSON file generated in the first step and browse the application. A small guide for its usage can be found here: [guide.pdf]. 
