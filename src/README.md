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
cd stelar-ner-graph/delivered
```

2. Build the Docker image:
```bash
docker build -t stelar-ner-dashboard .
```

## Running the Dashboard

1. Run the Docker container:
```bash
docker run -p 8501:8501 stelar-ner-dashboard
```

2. Access the dashboard:
Open your web browser and navigate to:
```
http://localhost:8501
```

## Features

The dashboard provides the following capabilities:

- Text input and analysis
- Named Entity Recognition (NER) using either LLMs or iFoodRoBERTa
- Entity linking with various methods (LLM, ChromaDB, BM25)
- Text summarization
- Main entity selection
- Results comparison and saving

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

## Development

The project structure is as follows:
```
.
├── Dashboard.py          # Main Streamlit application
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
