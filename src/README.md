# STELAR NER Graph Dashboard

This is a Streamlit-based dashboard for the STELAR NER Graph tool, which provides Named Entity Recognition (NER) and Entity Linking capabilities.

## Prerequisites

Before you begin, ensure you have the following installed:
- Docker
- Git (optional, for cloning the repository)

## Environment Configuration

1. Configure your environment variables:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and set your configuration values:
     - API keys
     - Model endpoints
     - Other environment-specific settings

## Building the Docker Image

1. Navigate to the project directory:
```bash
cd GenericNER/src
```

2. Build the Docker image:
```bash
docker build -t stelar-ner-dashboard .
```

## Running the Dashboard

1. Run the Docker container:
```bash
docker run -p 8502:8502 --env-file .env --env OLLAMA_HOST=http://host.docker.internal:11434 stelar-ner-dashboard
```

2. Access the dashboard:
Open your web browser and navigate to:
```
http://localhost:8502
```

## Features

The dashboard provides the following capabilities:

- Text input and analysis
- Named Entity Recognition (NER) using either LLMs or iFoodRoBERTa
- Entity linking with various methods (LLM, ChromaDB, BM25)
- Text summarization
- Main entity selection
- Results comparison and saving
- **Batch processing** for analyzing multiple texts at once

## Configuration Options

The dashboard can be configured through the sidebar with the following options:

### Summarization
- Enable/disable summarization
- Select summarization model

### Main Entity Selection
- No selection
- Single entity selection
- Multiple entity selection
- Model selection for entity selection

### Named Entity Recognition
- LLM-based NER
- iFoodRoBERTa NER
- Model selection for LLM-based NER

### Entity Linking
- Enable/disable entity linking
- Upload custom ontology
- Select linking method (LLM, ChromaDB, BM25)
- Model selection for LLM-based linking

## Batch Processing

The STELAR NER Graph tool supports batch processing of multiple texts through the `batch_run.py` module, allowing you to process large datasets efficiently.

### Basic Usage

```python
import pandas as pd
from archive.batch_run import batch_run_ner, create_default_parameters

# Load your data
df = pd.read_csv('your_data.csv')

# Create parameters for the pipeline
parameters = create_default_parameters()
parameters['class_of_interest'] = 'food'  # or any entity type you're interested in
parameters['ontology'] = ['food', 'beverage', 'ingredient', 'dish']

# Run batch processing
results_df = batch_run_ner(
    df=df,
    text_column='text',  # name of the column containing your texts
    parameters=parameters
)

# Save results
results_df.to_csv('results.csv', index=False)
```

### Input Data Format

Your input DataFrame should contain at least one column with the texts to be processed:

```csv
text
"The quick brown fox jumps over the lazy dog"
"I love eating pizza and drinking coffee"
"Fresh apples and organic vegetables are healthy"
```

### Parameters Configuration

The `create_default_parameters()` function provides sensible defaults, but you can customize any parameter:

```python
parameters = {
    "translation_option": False,           # Enable/disable translation
    "ontology": ["food", "beverage"],     # Custom ontology for entity linking
    "summary_option": True,               # Enable/disable summarization
    "summary_model": "ollama:llama3.1:latest",
    "main_entity_selection_option": True, # Enable main entity selection
    "main_entity_selection_type": "single", # "single" or "multiple"
    "main_entity_model": "ollama:llama3.1:latest",
    "ner_method": "LLM",                  # "LLM" or "iFoodRoBERTa"
    "ner_model": "ollama:llama3.1:latest",
    "entity_linking_method": "LLM",       # "LLM", "ChromaDB", or "BM25"
    "entity_linking_k": 3,                # Number of top matches to return
    "entity_linking_model": "ollama:llama3.1:latest",
    "class_of_interest": "food",          # Entity type to extract
    "evaluation_option": False
}
```

### Output Format

The batch processing returns a DataFrame with the original data plus additional columns:

- `annotations`: List of extracted entities
- `linked_annotations`: Dictionary mapping entities to ontology matches
- `summary`: Text summary (if enabled)
- `main_entities`: Selected main entities (if enabled)

### Custom Output Columns

You can customize the output column names:

```python
custom_output = {
    'annotations': 'extracted_entities',
    'linked_annotations': 'entity_links',
    'summary': 'text_summary',
    'main_entities': 'main_entity_list'
}

results_df = batch_run_ner(
    df=df,
    text_column='text',
    parameters=parameters,
    output_columns=custom_output
)
```

### Batch Processing with Evaluation

If you have ground truth annotations, you can evaluate the performance:

```python
from archive.batch_run import batch_run_with_evaluation

# Your DataFrame should have both text and ground truth columns
df_with_ground_truth = pd.read_csv('data_with_annotations.csv')

results = batch_run_with_evaluation(
    df=df_with_ground_truth,
    text_column='text',
    gold_standard_column='ground_truth_entities',
    parameters=parameters,
    prediction_column='predicted_entities'
)

# Access results and metrics
results_df = results['results_df']
metrics = results['evaluation_metrics']

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

### Error Handling

The batch processing includes error handling. If processing fails for a specific row, the error will be logged and the output columns for that row will contain error messages starting with "ERROR:".

### Performance Tips

1. **Model Selection**: Use faster models (e.g., smaller LLMs) for large batches
2. **Batch Size**: Process data in chunks if working with very large datasets
3. **Parallel Processing**: The current implementation processes rows sequentially; consider implementing parallel processing for better performance
4. **Resource Management**: Monitor memory usage when processing large datasets

### Example: Complete Workflow

```python
import pandas as pd
from archive.batch_run import batch_run_ner, create_default_parameters

# 1. Load and prepare data
df = pd.read_csv('documents.csv')
print(f"Processing {len(df)} documents...")

# 2. Configure parameters
parameters = create_default_parameters()
parameters.update({
    'class_of_interest': 'food',
    'ontology': ['fruit', 'vegetable', 'meat', 'dairy', 'grain'],
    'summary_option': True,
    'main_entity_selection_option': True,
    'entity_linking_k': 5
})

# 3. Run batch processing
results_df = batch_run_ner(
    df=df,
    text_column='document_text',
    parameters=parameters
)

# 4. Analyze results
print(f"Processing complete. Results shape: {results_df.shape}")
print(f"Columns: {list(results_df.columns)}")

# 5. Save results
results_df.to_csv('ner_results.csv', index=False)
print("Results saved to ner_results.csv")
```

## Development

The project structure is as follows:
```
.
├── Dashboard.py          # Main Streamlit application
├── stelar_graph.py      # Core NER pipeline graph
├── archive/
│   ├── batch_run.py     # Batch processing functionality
│   ├── evaluation.py    # Evaluation metrics
│   └── functions.py     # Core NER functions
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
├── .env.example        # Example environment configuration
├── .env                # Your environment configuration (create from .env.example)
└── README.md           # This file
```

## Dependencies

The main dependencies include:
- streamlit
- pandas
- langchain
- transformers
- torch
- and other supporting libraries

All dependencies are listed in `requirements.txt` and will be automatically installed when building the Docker image.

## Troubleshooting

If you encounter any issues:

1. Ensure Docker is running
2. Check if port 8501 is available
3. Verify that all dependencies are properly installed
4. Check the Docker logs for any error messages
5. Ensure your `.env` file is properly configured

## LangGraph Studio Execution 

### Installation

1. Ensure Docker is installed:
   ```bash
   docker --version
   ```

2. Install the CLI package:

   **Python:**
   ```bash
   pip install langgraph-cli
   ```

   **JavaScript:**
   ```bash
   npm install @langchain/langgraph-cli
   ```

3. Verify installation:
   ```bash
   # Python
   langgraph --help
   
   # JavaScript
   npx @langchain/langgraph-cli --help
   ```

For more information, visit the [LangGraph CLI documentation](https://langchain-ai.github.io/langgraph/cloud/reference/cli/).

### Execute the Graph

1. Navigate to the directory containing `langgraph.json`:
   ```bash
   cd path/to/langgraph.json
   ```

2. Start the LangGraph development server:
   ```bash
   langgraph dev
   ```

3. Access the interface:
   - GUI: Open [LangGraph Studio](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024) in your browser
   - API: Use the endpoint at `http://127.0.0.1:2024`

For detailed information about LangGraph's features and usage, refer to the [official documentation](https://langchain-ai.github.io/langgraph/).

## Support

For any issues or questions, please contact the STELAR project team.
