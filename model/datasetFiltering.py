# import logging
from fuzzywuzzy import fuzz
import pandas

# logger = logging.getLogger(__name__)

# input_path = task_definition['input']['unload_dataset']['final_path']
input_path = '/Users/ranziv/Downloads/cem-output/unload_output/unload_results.csv'
# output_path = task_definition['output']['filter_dataset']['final_path']
output_path = '/Users/ranziv/Downloads/cem-output/datasetFiltering_output-no-fuzzy/'
df = pandas.read_csv(input_path)
scores_list=[]
for index, row in df.iterrows():
    scores_list.append(fuzz.partial_ratio(str(row['a']), str(row['b'])))
df['score']=scores_list
print(df.head(10))
# df = df[df.score > 50]
print(df.head(10))
df[['a', 'b']].to_parquet(f'{output_path}/filtered.parquet')
df[['a', 'b']].to_csv(output_path+'filtered.csv', compression='infer', index=False)
# df[['a', 'b', 'score']].to_csv(output_path+'filtered.csv', compression='infer', index=False)
