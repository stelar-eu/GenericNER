import pandas as pd
from minio import Minio
from entity_extraction import entity_extraction
from backend_functions import make_df_by_argument
from llm_foodNER_functions import prepare_dataset_new
import json
import uuid
from time import time
import sys
import traceback

def run(j):
    try:
        outfile, log = "", {}
        inputs = j['input']
        INPUT_FILE = inputs[0]
        # File Parameters
        output_file = j['parameters']['output_file']
        text_column = j['parameters']['text_column']
        ground_truth_column = j['parameters'].get('ground_truth_column')
        product_column = j['parameters'].get('product_column')
        csv_delimiter = j['parameters']['csv_delimiter']
        keep_food = j['parameters'].get('keep_food', False)
        
        #Algorithm Parameters
        N = j['parameters']['N']
        #extraction_type = j['parameters']['extraction_type']
        #model = j['parameters']['model']
        prediction_values = j['parameters']['prediction_values']
        syntactic_analysis_tool = j['parameters'].get('syntactic_analysis_tool')
        if syntactic_analysis_tool is None:
            syntactic_analysis_tool = 'stanza'
        prompt_id = j['parameters'].get('prompt_id')
        if prompt_id is None:
            prompt_id = 0
        minio = j['minio']
        
        df = prepare_dataset_new(INPUT_FILE, text_column = text_column, ground_truth_column = ground_truth_column,
                                 product_column = product_column, csv_delimiter=csv_delimiter, minio = minio)

        t = time()
        outfile, log = entity_extraction(df, prediction_values,
                                         output_file=output_file, N = N,
                                         syntactic_analysis_tool = syntactic_analysis_tool, 
                                         prompt_id = prompt_id)
        t = time() - t
        # log = {k: float(v[:-1]) for k, v in log['score'].items()}

        client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
        
        if keep_food: # Keep only FOOD Tags
            #outfile2 = outfile.replace('.csv', '_food.csv')
            df = pd.read_csv(outfile+'.csv')
            df_stats = pd.DataFrame()
            df_stats['total_tags'] = df.groupby('text_id')['phrase_id'].count()
            df = df.loc[df.tag=='FOOD']
            df_stats['food_tags'] = df.groupby('text_id')['phrase_id'].count()
            log = df_stats.mean().to_dict()
            log['execution_time'] = t
            df.to_csv(outfile+'.csv', index=False, header=True)
        
        basename = str(uuid.uuid4()) + ".csv"
        result = client.fput_object(minio['bucket'], basename, outfile+".csv")
        object_path = f"s3://{result.bucket_name}/{result.object_name}"
        
        basename_2 = str(uuid.uuid4()) + ".json"
        result_2 = client.fput_object(minio['bucket'], basename_2, outfile+".json")
        object_path_2 = f"s3://{result_2.bucket_name}/{result_2.object_name}"


        return {'message': 'Entity Extraction completed successfully!',
                'output': [{"path": object_path, "name": "Extracted Entities in CSV format"},
                           {"path": object_path_2, "name": "Extracted Entities in JSON format"}
                           ],
                'metrics': log,
                'status': 200}
                        #'output': [object_path2], 'metrics': log})
    except Exception as e:
        return {
            'error': traceback.format_exc(),
            'message': 'An error occurred during data processing.',
            'status': 500
        }
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError("Please provide 2 files.")
    with open(sys.argv[1]) as o:
        j = json.load(o)
    response = run(j)
    with open(sys.argv[2], 'w') as o:
        o.write(json.dumps(response, indent=4))
