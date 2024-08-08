# Extraction & Deduplication Evaluation Dashboard [Optional]

The purpose of this dashboard is to help evaluate the results produced from the NER and the deduplication components.

The directory contains all the necessary code to run the app, as well as two dataframes `annotations_df_expanded.csv` and `canditate_pairs_no_dups_augmented.csv`, which contain the named entity incident annotations and the deduplication canditates respectively and act as main datasource for the dashboard. The user can replace these data with other data compliant to their format or use our `connector` (🚧 Currently under construction🚧)   

## Instructions 

The app is built in such a way that it can be deployed easily with `docker`. To do so the user needs to have `docker` installed navigate to the current directory and execute the command:

```shell
docker build -t eval_dash .
```

After succesfully building the container the user should run it using the following command:

```shell 
docker run -d eval_dash
```

The dashboard can then be accessed through normal browsers by navigating to: 

```url
localhost:8000
```

Currently the dashboard is built in a way to support **single-user usage**. When the user completes their work within the app they cann retrieve the results in .csv format by copying the .csv files to a local directory in the following way:

```shell
docker cp <container_id>:/app/annotations_df_expanded.csv /path/to/destination/on/host/annotations_df_expanded.csv
```
and 
```shell
docker cp <container_id>:/app/canditate_pairs_no_dups_augmented.csv /path/to/destination/on/host/annotations_df_expanded.csv
```

The results are present as new columns into the initial dataframes.