In order to visualize and compare entity labels identified by various NER models in the NER component, we developed a web-based graphical user interface (GUI). 
Implemented in Python using [Streamlit] package, this GUI allows the user to interactively inspect the entities identified by each model. The user can choose to print a text based on multiple filters (e.g., which sentences contain LOC entities) or view annotations only for specific labels of their choice (e.g., only locations or organizations or only annotations by spaCy). The user can also display various statistics, i.e., how many different entities were detected by each tool as well as evaluation metrics, if ground truth is available.