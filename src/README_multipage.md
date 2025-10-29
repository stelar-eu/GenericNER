# STELAR NER-EL Multi-Page Streamlit App

This directory contains a multi-page Streamlit application for the STELAR Named Entity Recognition and Entity Linking system.

## Structure

```
src/
├── app.py                          # Main application entry point
├── pages/
│   ├── __init__.py                # Pages package initialization
│   ├── dashboard.py               # Main dashboard for running NER-EL pipeline
│   └── evaluation.py              # Evaluation page for inspecting results
├── sample_evaluation_data.csv     # Sample CSV data for evaluation page
├── sample_metrics.json           # Sample JSON metrics for evaluation page
└── README_multipage.md           # This file
```

## Running the Application

To run the multi-page Streamlit app:

```bash
cd src/
streamlit run app.py
```

## Pages

### 1. Dashboard Page
- **Purpose**: Interactive interface for configuring and running the NER-EL pipeline
- **Features**:
  - Pipeline configuration (translation, summarization, NER, entity linking)
  - Real-time text analysis
  - Results visualization with annotated text
  - Save and compare results
- **Usage**: Enter text, configure pipeline settings in sidebar, click "Analyze"

### 2. Evaluation Page
- **Purpose**: Load and inspect evaluation datasets with ground truth annotations
- **Features**:
  - Upload CSV files with evaluation data
  - Upload JSON files with overall metrics
  - Row-by-row inspection of annotations
  - Side-by-side comparison of ground truth vs model predictions
  - Calculate per-row metrics (precision, recall, F1-score)
  - Visualize annotations with color coding

## Data Formats

### CSV File Format (for Evaluation Page)
The CSV file should contain the following columns:
- `text`: The original text to be analyzed
- `ground_truth_annotations`: Ground truth entities (as list or comma-separated string)
- `model_annotations`: Model predicted entities (as list or comma-separated string)
- `linked_annotations`: Entity linking results (as dictionary string)
- `class_of_interest`: The entity class being evaluated

### JSON File Format (for Metrics)
The JSON file should contain evaluation metrics such as:
```json
{
    "overall_precision": 0.847,
    "overall_recall": 0.792,
    "overall_f1_score": 0.819,
    "exact_match_accuracy": 0.654,
    "total_samples": 150,
    "model_name": "llama3.1:latest"
}
```

## Sample Data

Sample files are provided:
- `sample_evaluation_data.csv`: Example evaluation dataset
- `sample_metrics.json`: Example metrics file

## Dependencies

Make sure you have the required dependencies installed:
```bash
pip install streamlit streamlit-extras pandas
```

## Navigation

Use the sidebar navigation to switch between:
- **Dashboard**: Main NER-EL pipeline interface
- **Evaluation**: Evaluation data inspection and analysis

## Features

### Dashboard Features
- Configure translation, summarization, NER, and entity linking options
- Upload custom ontologies for entity linking
- Real-time pipeline execution
- Annotated text visualization
- Results comparison interface

### Evaluation Features
- Load evaluation datasets from CSV files
- Load metrics from JSON files
- Detailed row-by-row inspection
- Color-coded annotation display (green for ground truth, red for predictions)
- Per-row performance metrics calculation
- Side-by-side comparison view

## Tips

1. **File Uploads**: Use the file uploaders in the evaluation page to load your data
2. **Row Selection**: Use the dropdown to select specific rows for detailed inspection
3. **Annotation Parsing**: The app automatically handles different annotation formats (lists, strings)
4. **Color Coding**: Green highlights indicate ground truth, red indicates model predictions
5. **Metrics**: Overall metrics are displayed at the top, per-row metrics in the comparison tab
