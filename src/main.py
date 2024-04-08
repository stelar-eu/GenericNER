from entity_linking import  prep_df
from llm_foodNER_functions import prepare_dataset_new, generate_output_file_name, read_configuration_file, read_configuration_file_ont
from entity_extraction import entity_extraction

def main():
    dataset, text_column, ground_truth_column, product_column, csv_delimiter, prediction_values, N, minio = read_configuration_file('../config_file.ini')     
    output_file = generate_output_file_name(dataset,prediction_values)
    df = prepare_dataset_new(dataset, text_column = text_column, ground_truth_column = ground_truth_column,
                             product_column = product_column, csv_delimiter=csv_delimiter, minio = minio)
    if df.empty:
     return -1
    ontology_file, ontology_header, ontology_col_id, ontology_col_text, ontology_col_separator, ontology_text_separator, delta_alg, similarity = read_configuration_file_ont('../config_file.ini')
    if ontology_file != None:
     ontology_df = prep_df(ontology_file, ontology_header, ontology_col_text, ontology_col_separator, ontology_text_separator, minio)
    else:
     ontology_df = None
    outfile, log = entity_extraction(df, prediction_values,
                                     output_file=output_file, N = N,
                                    ontology_df = ontology_df, 
                                    ontology_col_id=ontology_col_id, 
                                    ontology_col_text=ontology_col_text,
                                    similarity=similarity, k=1, delta_alg=delta_alg)
    print(log)
    
if __name__ == "__main__":
  main()    
