import fsspec
import json
from random import sample
import os
import pickle

def random_sample( data , sample_size ):
    res_dict = {}
    res_dict['company_list'] = sample(data['company_list'],sample_size)
    return res_dict

company_data = '/Users/ranziv/Downloads/data-f1k.json'
sample_path = '/Users/ranziv/Downloads/cem-input/data-f1k-same-samples/'
sample_size = 926
amount_of_samples = 1

os.system('rm -rf ' + sample_path)
os.system('mkdir ' + sample_path)

with fsspec.open(company_data) as company_data_fh:
    data = json.load(company_data_fh)

for i in range(1, amount_of_samples+1):
    fout = sample_path + 'data-f1k-sample-' + str(i) + '.json'
    sample_file = open(fout, 'wb')
    sample_data = random_sample(data , sample_size)
    pickle.dump(sample_data, sample_file, protocol=pickle.HIGHEST_PROTOCOL)
    sample_file.close()

    fin = open(fout, 'rb')
    test = pickle.load(fin)
    print(test)
#
# fin = open('/Users/ranziv/Downloads/cem-input/data-f1k-samples/data-f1k-sample-11.json', 'rb')
# test = pickle.load(fin)
# print(test)