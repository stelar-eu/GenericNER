import streamlit as st
import pandas as pd
import json
import ast
from streamlit_extras.annotated_text import annotated_text

def show():
    st.title("Evaluation Dashboard")
    st.markdown("""
    This page allows you to load evaluation data from CSV and JSON files to inspect model performance 
    and compare annotations with ground truth data.
    """)
    
    # File upload section
    st.header("Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Evaluation Data (CSV)")
        csv_file = st.file_uploader(
            "Upload CSV file with texts, ground truth annotations, model annotations, and linked annotations", 
            type="csv",
            key="eval_csv"
        )
        
    with col2:
        st.subheader("Upload Metrics (JSON)")
        json_file = st.file_uploader(
            "Upload JSON file with overall evaluation metrics", 
            type="json",
            key="eval_json"
        )
    
    # Initialize session state for evaluation data
    if 'eval_df' not in st.session_state:
        st.session_state.eval_df = None
    if 'eval_metrics' not in st.session_state:
        st.session_state.eval_metrics = None
    
    # Load CSV data
    if csv_file is not None:
        try:
            st.session_state.eval_df = pd.read_csv(csv_file)
            st.success(f"Successfully loaded CSV with {len(st.session_state.eval_df)} rows")
            
            # Display column information
            st.write("**Columns in the dataset:**")
            st.write(list(st.session_state.eval_df.columns))
            
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
    
    # Load JSON data
    if json_file is not None:
        try:
            st.session_state.eval_metrics = json.load(json_file)
            st.success("Successfully loaded metrics JSON")
        except Exception as e:
            st.error(f"Error loading JSON file: {e}")
    
    # Display metrics if available
    if st.session_state.eval_metrics is not None:
        st.header("Overall Metrics")
        
        # Organize metrics into logical groups
        performance_metrics = {}
        count_metrics = {}
        metadata = {}
        
        for metric_name, metric_value in st.session_state.eval_metrics.items():
            if any(perf in metric_name.lower() for perf in ['precision', 'recall', 'f1', 'accuracy']):
                performance_metrics[metric_name] = metric_value
            elif any(count in metric_name.lower() for count in ['total', 'true_positive', 'false_positive', 'false_negative', 'average']):
                count_metrics[metric_name] = metric_value
            else:
                metadata[metric_name] = metric_value
        
        # Display performance metrics (first row)
        if performance_metrics:
            st.subheader("Performance Metrics")
            perf_cols = st.columns(min(4, len(performance_metrics)))
            for i, (metric_name, metric_value) in enumerate(performance_metrics.items()):
                with perf_cols[i % 4]:
                    if isinstance(metric_value, (int, float)):
                        st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.3f}")
                    else:
                        st.metric(metric_name.replace('_', ' ').title(), str(metric_value))
        
        # Display count metrics (second row)
        if count_metrics:
            st.subheader("Count & Statistical Metrics")
            count_cols = st.columns(min(4, len(count_metrics)))
            for i, (metric_name, metric_value) in enumerate(count_metrics.items()):
                with count_cols[i % 4]:
                    if isinstance(metric_value, (int, float)):
                        if 'average' in metric_name.lower():
                            st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.2f}")
                        else:
                            st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:,}")
                    else:
                        st.metric(metric_name.replace('_', ' ').title(), str(metric_value))
        
        # Display metadata (third row)
        if metadata:
            st.subheader("Metadata")
            meta_cols = st.columns(min(4, len(metadata)))
            for i, (metric_name, metric_value) in enumerate(metadata.items()):
                with meta_cols[i % 4]:
                    st.metric(metric_name.replace('_', ' ').title(), str(metric_value))
        
        # Display detailed metrics
        st.subheader("Detailed Metrics")
        st.json(st.session_state.eval_metrics)
    
    # Display and inspect evaluation data
    if st.session_state.eval_df is not None:
        st.header("Evaluation Data")
        
        # Show basic statistics
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(st.session_state.eval_df))
        with col2:
            if 'text' in st.session_state.eval_df.columns:
                avg_length = st.session_state.eval_df['text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_length:.0f} chars")
        with col3:
            if 'ground_truth_annotations' in st.session_state.eval_df.columns:
                # Try to count annotations (assuming they're stored as lists or strings)
                try:
                    def count_annotations(ann_str):
                        if pd.isna(ann_str):
                            return 0
                        if isinstance(ann_str, str):
                            try:
                                ann_list = ast.literal_eval(ann_str)
                                return len(ann_list) if isinstance(ann_list, list) else 0
                            except:
                                return len(ann_str.split(',')) if ann_str else 0
                        elif isinstance(ann_str, list):
                            return len(ann_str)
                        return 0
                    
                    avg_annotations = st.session_state.eval_df['ground_truth_annotations'].apply(count_annotations).mean()
                    st.metric("Avg GT Annotations", f"{avg_annotations:.1f}")
                except:
                    st.metric("Avg GT Annotations", "N/A")
        
        # Display the dataframe
        st.subheader("Data Preview")
        st.dataframe(st.session_state.eval_df.head(10))
        
        # Row selection for detailed inspection
        st.header("Detailed Inspection")
        
        if len(st.session_state.eval_df) > 0:
            selected_index = st.selectbox(
                "Select a row to inspect in detail:",
                options=range(len(st.session_state.eval_df)),
                format_func=lambda x: f"Row {x}: {st.session_state.eval_df.iloc[x].get('text', 'N/A')[:50]}..."
            )
            
            if selected_index is not None:
                selected_row = st.session_state.eval_df.iloc[selected_index]
                
                # Display selected row details
                st.subheader(f"Detailed View - Row {selected_index}")
                
                # Text display
                if 'text' in selected_row:
                    st.write("**Original Text:**")
                    st.write(selected_row['text'])
                
                # Create tabs for different annotation types
                tab1, tab2, tab3, tab4 = st.tabs(["Ground Truth", "Model Predictions", "Linked Annotations", "Comparison"])
                
                with tab1:
                    st.subheader("Ground Truth Annotations")
                    if 'ground_truth_annotations' in selected_row:
                        gt_annotations = selected_row['ground_truth_annotations']
                        
                        # Try to parse annotations if they're stored as strings
                        if isinstance(gt_annotations, str) and gt_annotations:
                            try:
                                gt_annotations = ast.literal_eval(gt_annotations)
                            except:
                                gt_annotations = gt_annotations.split(',') if gt_annotations else []
                        
                        if gt_annotations and 'text' in selected_row:
                            formatted_gt = format_annotations_for_display(
                                selected_row['text'], 
                                gt_annotations, 
                                "Ground Truth", 
                                "#00ff00"
                            )
                            annotated_text(*formatted_gt)
                        
                        st.write("**Ground Truth Entities:**")
                        st.write(gt_annotations)
                
                with tab2:
                    st.subheader("Model Predictions")
                    if 'model_annotations' in selected_row:
                        model_annotations = selected_row['model_annotations']
                        
                        # Try to parse annotations if they're stored as strings
                        if isinstance(model_annotations, str) and model_annotations:
                            try:
                                model_annotations = ast.literal_eval(model_annotations)
                            except:
                                model_annotations = model_annotations.split(',') if model_annotations else []
                        
                        if model_annotations and 'text' in selected_row:
                            formatted_model = format_annotations_for_display(
                                selected_row['text'], 
                                model_annotations, 
                                "Model Prediction", 
                                "#ff0000"
                            )
                            annotated_text(*formatted_model)
                        
                        st.write("**Model Predicted Entities:**")
                        st.write(model_annotations)
                
                with tab3:
                    st.subheader("Linked Annotations")
                    if 'linked_annotations' in selected_row:
                        linked_annotations = selected_row['linked_annotations']
                        
                        # Try to parse linked annotations if they're stored as strings
                        if isinstance(linked_annotations, str) and linked_annotations:
                            try:
                                linked_annotations = ast.literal_eval(linked_annotations)
                            except:
                                linked_annotations = {}
                        
                        st.write("**Entity Linking Results:**")
                        if isinstance(linked_annotations, dict):
                            for entity, links in linked_annotations.items():
                                st.write(f"**{entity}:** {links}")
                        else:
                            st.write(linked_annotations)
                
                with tab4:
                    st.subheader("Comparison")
                    
                    # Side-by-side comparison
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.write("**Ground Truth**")
                        if 'ground_truth_annotations' in selected_row:
                            gt_annotations = selected_row['ground_truth_annotations']
                            if isinstance(gt_annotations, str) and gt_annotations:
                                try:
                                    gt_annotations = ast.literal_eval(gt_annotations)
                                except:
                                    gt_annotations = gt_annotations.split(',') if gt_annotations else []
                            
                            if gt_annotations and 'text' in selected_row:
                                formatted_gt = format_annotations_for_display(
                                    selected_row['text'], 
                                    gt_annotations, 
                                    "GT", 
                                    "#00ff00"
                                )
                                annotated_text(*formatted_gt)
                    
                    with comp_col2:
                        st.write("**Model Predictions**")
                        if 'model_annotations' in selected_row:
                            model_annotations = selected_row['model_annotations']
                            if isinstance(model_annotations, str) and model_annotations:
                                try:
                                    model_annotations = ast.literal_eval(model_annotations)
                                except:
                                    model_annotations = model_annotations.split(',') if model_annotations else []
                            
                            if model_annotations and 'text' in selected_row:
                                formatted_model = format_annotations_for_display(
                                    selected_row['text'], 
                                    model_annotations, 
                                    "Model", 
                                    "#ff0000"
                                )
                                annotated_text(*formatted_model)
                    
                    # Calculate and display metrics for this specific row
                    if 'ground_truth_annotations' in selected_row and 'model_annotations' in selected_row:
                        gt_set = set(gt_annotations) if gt_annotations else set()
                        model_set = set(model_annotations) if model_annotations else set()
                        
                        if gt_set or model_set:
                            true_positives = len(gt_set.intersection(model_set))
                            false_positives = len(model_set - gt_set)
                            false_negatives = len(gt_set - model_set)
                            
                            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            
                            st.write("**Row-level Metrics:**")
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Precision", f"{precision:.3f}")
                            with metric_col2:
                                st.metric("Recall", f"{recall:.3f}")
                            with metric_col3:
                                st.metric("F1-Score", f"{f1:.3f}")
                
                # Display all row data
                st.subheader("Complete Row Data")
                st.json(selected_row.to_dict())

def format_annotations_for_display(text, entities, entity_tag, color="#ff0000"):
    """Format annotations for display with streamlit_extras.annotated_text"""
    if not entities or not text:
        return [text] if text else []
    
    # Ensure entities is a list
    if isinstance(entities, str):
        entities = [entities]
    
    # Sort entities by their position in the text to maintain order
    entity_positions = []
    for entity in entities:
        if entity and entity.strip():  # Skip empty entities
            start = text.find(entity.strip())
            if start != -1:
                entity_positions.append((start, entity.strip()))
    
    entity_positions.sort(key=lambda x: x[0])
    
    annotations = []
    last_end = 0
    
    for start, entity in entity_positions:
        # Add the text before the entity
        if start > last_end:
            annotations.append(text[last_end:start])
            
        # Add the entity with its annotation
        annotations.append((entity, entity_tag, color))
        
        last_end = start + len(entity)
    
    # Add any remaining text after the last entity
    if last_end < len(text):
        annotations.append(text[last_end:])
        
    return annotations
