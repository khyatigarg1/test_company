import json
import gzip
import binascii
import os
import copy
import datetime
import fingerprint
import hashlib



execution_dt = str(datetime.datetime.now()).replace(' ','_').replace('-','').replace(':','')
part_size = 20000
min_record = 0
max_record = 1000000
fprint = fingerprint.Fingerprint(kgram_len=4, window_len=5, base=10, modulo=1000)
# fprint = fingerprint.Fingerprint(kgram_len=50, window_len=100, base=101, modulo=sys.maxsize)

fname = '/Users/ranziv/Downloads/corpus_snapshot-2020-06-09.jpl.gz'
#fname = '/home/ranziv/jpl/corpus_snapshot-2020-06-09.jpl.gz'
#fname = '/Users/ranziv/Downloads/shard-cb_active.jpl.gz'

ofname_prefix = '/Users/ranziv/Downloads/corpus_snapshot-2020-06-09----ELDAN'
#ofname = '/Users/ranziv/Downloads/corpus_snapshot-2020-06-09'+execution_dt+'.json.gz'
#ofname = '/home/ranziv/jpl/corpus_snapshot-2020-06-09'+execution_dt+'.json.gz'



part = 1
i = 0
f = gzip.open(fname, 'r')
ofname = ofname_prefix + execution_dt + '_part_' + str(part) + '.json.gz'
print(str(datetime.datetime.now()) + 'opening file ' + ofname)
gz = gzip.GzipFile(ofname, 'wb', 9)

for line in f:
    if min_record <= i and i < max_record:
        i = i + 1
        if i % part_size == 0:
            print(str(datetime.datetime.now()) + 'closing file ' + ofname)
            gz.close()
            part = part + 1
            ofname = ofname_prefix + execution_dt + '_part_' + str(part) + '.json.gz'
            print(str(datetime.datetime.now()) + 'opening file ' + ofname)
            gz = gzip.GzipFile(ofname, 'wb', 9)
        if i % 10000 == 0:
                print('------------------------- ' + str(i) + ' -------------------------')
                print(datetime.datetime.now())

        my_dict={}
        json_content = json.loads(line)
        # for item in json_content:
        my_dict['raw_title'] = copy.deepcopy(json_content['raw_title'])
        my_dict['resolved_city'] = copy.deepcopy(json_content['resolved_city'])
        my_dict['resolved_state_code'] = copy.deepcopy(json_content['resolved_state_code'])
        my_dict['resolved_country_code'] = copy.deepcopy(json_content['resolved_country_code'])
        my_dict['org_name'] = copy.deepcopy(json_content['org_name'])
        my_dict['fingerprint'] = copy.deepcopy(json_content['fingerprint'])
        my_dict['open_seat_id'] = copy.deepcopy(json_content['open_seat_id'])
        my_dict['posted_time'] = copy.deepcopy(json_content['posted_time'])
        my_dict['description'] = copy.deepcopy(json_content['description'])

        try:
            fingerprint = fprint.generate(my_dict['description'])
            hash_object = hashlib.md5(str(fingerprint).encode())
            my_dict['winnowing_md5'] = hash_object.hexdigest()
        except:
            pass
            print("error fingerprinting")

        # print(my_dict['winnowing_md5'])



        back_json = json.dumps(my_dict)


        gz.write((back_json+'\n').encode())
        gz.flush()
    else: break

print(str(datetime.datetime.now()) + 'closing file ' + ofname)
gz.close()
f.close()
