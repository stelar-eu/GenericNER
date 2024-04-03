import pandas as pd
# import pytokenjoin as ptj
from minio import Minio
import importlib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
import traceback

def prep_df(input_file, header, col_text, col_separator, text_separator, minio=None):
    """
    Prepare DataFrame from input file.
    """
    
    header = header if header != -1 else None
    
    if input_file.startswith('s3://'):
        bucket, key = input_file.replace('s3://', '').split('/', 1)
        client = Minio(minio['endpoint_url'], access_key=minio['id'], secret_key=minio['key'])
        df = pd.read_csv(client.get_object(bucket, key), header=header, sep=col_separator, on_bad_lines = 'warn')
    else:
        df = pd.read_csv(input_file, header=header, sep=col_separator, on_bad_lines = 'warn')
    df.columns = [str(c) for c in df.columns]
    
    df[f"{col_text}_original"] = df[col_text].copy()
    df[col_text] = df[col_text].str.split(text_separator)
    df = df.loc[~(df[col_text].isna())]
    df[col_text] = df[col_text].apply(lambda x: list(set(x)))
    
    df.columns = [str(col) for col in df.columns]
    return df

def entity_linking(df_left, col_id_left, col_text_left,  col_ground_left,
                   df_right, col_id_right, col_text_right,
                   similarity, k, delta_alg):
    try:
        if similarity not in ['jaccard', 'edit']:
            raise ValueError("Similarity must be in ['jaccard', 'edit']")        
            
        module = importlib.import_module('pytokenjoin.' + similarity + '.join_knn')
        module = module.TokenJoin()
        
        left_attr = [f"{col_text_left}_original"]
        if col_ground_left is not None:
            left_attr.append(col_ground_left)
        right_attr = [f"{col_text_right}_original"]
        
        pairs, log = module.tokenjoin_foreign(df_left, df_right, 
                                              col_id_left, col_id_right,
                                              col_text_left, col_text_right,
                                              k=k, delta_alg=delta_alg, keepLog=True,
                                              left_attr=left_attr, right_attr=right_attr)
            
        if col_ground_left is not None:
            mlb = MultiLabelBinarizer()
            y_true_bin = mlb.fit_transform(pairs[f"l_{col_ground_left}"])
            y_pred_bin = mlb.transform(pairs[f"r_{col_text_right}_original"])

            log = {'total_time': log['total_time']}
            for avg in ['micro', 'macro', 'weighted']:
                log[f'precision_{avg}'] = precision_score(y_true_bin, y_pred_bin, average=avg)
                log[f'recall_{avg}'] = recall_score(y_true_bin, y_pred_bin, average=avg)
                log[f'f1_{avg}'] = f1_score(y_true_bin, y_pred_bin, average=avg)
                
        return pairs, log
    except Exception as e:
        print(traceback.format_exc())
        return None, {}
