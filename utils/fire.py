
# fout = open('/Users/ranziv/Downloads/aliases-data.json','r')
# ss = fout.read()
# sset = eval(ss)
# print(sset)

import json
fout = open('/Users/ranziv/Downloads/data.json','r')
jdata = json.load(fout)

f1000 = open('/Users/ranziv/Downloads/f1000.csv','w')
synonyms_cnt = 0
canonical_cnt = 0
synonyms_buckets={}

for i in jdata['company_list']:
    line = i['canonical_company_name']
    subline = '"' + i['canonical_company_name'] + '"' + '>' + '"' + i['canonical_company_name'] + '"'
    print(subline)
    f1000.write(subline + '\n')
    synonyms_cnt = synonyms_cnt + 1
    canonical_cnt = canonical_cnt + 1
    for j in i['synonyms']:
        subline =  '"' + i['canonical_company_name'] + '"' + '>' + '"' + j + '"'
        line = line + '>' + j
        print(subline)
        f1000.write(subline + '\n')
        synonyms_cnt = synonyms_cnt + 1
    # print(line)
    if len(i['synonyms'])+1 in synonyms_buckets.keys():
        synonyms_buckets[len(i['synonyms'])+1] = synonyms_buckets[len(i['synonyms'])+1] + 1
    else:
        synonyms_buckets[len(i['synonyms'])+1] = 1
f1000.close()

print('canonical_cnt=' + str(canonical_cnt))
print('synonyms_cnt=' + str(synonyms_cnt))
print('average=' + str(synonyms_cnt/canonical_cnt))
print('buckets: ' + str(synonyms_buckets))



import matplotlib.pyplot as plt
#
# D = synonyms_buckets
#
# plt.bar(range(len(D)), list(D.values()), align='center')
# plt.xticks(range(52), list(map(int,D.keys())).sort())
#
# plt.show()


#
# data = synonyms_buckets
# names = list(data.keys())
# values = list(data.values())
# plt.bar(0,values[0],tick_label=names[0])
# plt.bar(1,values[1],tick_label=names[1])
# plt.bar(2,values[2],tick_label=names[3])
# plt.bar(3,values[2],tick_label=names[4])
# plt.bar(4,values[2],tick_label=names[5])
# plt.bar(2,values[2],tick_label=names[6])
# plt.bar(2,values[2],tick_label=names[7])
# plt.bar(2,values[2],tick_label=names[8])
# plt.bar(2,values[2],tick_label=names[9])
# plt.bar(2,values[2],tick_label=names[10])
# plt.bar(2,values[2],tick_label=names[11])
# plt.bar(2,values[2],tick_label=names[12])
#
# plt.xticks(range(0,3),names)
# # plt.savefig('fruit.png')
# plt.show()


a_dictionary = synonyms_buckets
keys = a_dictionary.keys()
values = a_dictionary.values()

plt.xlim([1, 20])
plt.bar(keys, values)
# for key in synonyms_buckets.keys().sort()
