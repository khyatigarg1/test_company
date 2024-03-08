import json
import gzip
import binascii
import os
import copy
import datetime
import fingerprint
import hashlib
import pyspark
from pyspark.sql.functions import current_date
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.6/bin/python3'
from pyspark import SparkConf
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

conf = SparkConf()
sc = pyspark.SparkContext(appName='mm_exp', conf=conf)
sqlContext = pyspark.SQLContext(sc)



fprint = fingerprint.Fingerprint(kgram_len=4, window_len=5, base=10, modulo=1000)
def fp_generation(s):
    try:
        fingerprint = fprint.generate(s)
        hash_object = hashlib.md5(str(fingerprint).encode())
        return hash_object.hexdigest()
    except:
        pass
        return ''

sqlContext.udf.register('fp_generation',
                        fp_generation,
                         StringType())

df = sqlContext.read.json('/Users/ranziv/Downloads/corpus_snapshot-2020-06-09.json.gz')
#df = sqlContext.read.json('/Users/ranziv/Downloads/corpus_snapshot-2020-06-0920200621_182243.305381_part_1.json.gz')
#df = sqlContext.read.json('/home/ranziv/jpl/corpus_snapshot-2020-06-0920200615_010953.062997.json.gz')
df.createGlobalTempView('df')
df_fp = sqlContext.sql("""
            SELECT fp_generation(description) as fp_md5, * from global_temp.df
            """).write.json("/Users/ranziv/Downloads/fp.json.gz.full/", compression="gzip")
    #/Users/ranziv/Downloads/fp.json.gz
    #/home/ranziv/jpl/output_fp/

