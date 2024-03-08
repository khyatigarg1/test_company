import nmslib
import os
import pickle
from scipy.sparse import issparse
import  numpy

class Index:
    """ class that encapsulates the compres model and ann index"""
    def __init__(self, model, inverted_index, aliases, index):
        self.index = index
        self.model = model
        self.inverted_index = inverted_index
        self.aliases = aliases

    @staticmethod
    def build_simple_index(model, dataset, use_aliases=True):
        company_names = {x: [x] for x in dataset}
        return Index.build_index(model, company_names)

    @staticmethod
    def build_compres_index(model, compres_data, use_aliases=True):
        company_names = {x['canonical_company_name']: [x['canonical_company_name']] + (x['synonyms'] if use_aliases else []) for x in compres_data['company_list']}
        return Index.build_index(model, company_names)

    @staticmethod
    def build_index(model, dataset):
        """
         dataset an dict of object => aliases
         object should be a simple type (typicaly id or name - must be json serializable), aliases should be a list of strings
         """
        inverted_index = {obj: alias for obj, aliases in dataset.items() for alias in aliases}
        aliases = list(inverted_index.keys())

        vectorized_dataset = model.predict(aliases)

        if issparse(vectorized_dataset):
            data_type = nmslib.DataType.SPARSE_VECTOR
            space = 'cosinesimil_sparse'
        else:
            data_type = nmslib.DataType.DENSE_VECTOR
            space = 'cosinesimil'

        # print(numpy.where((numpy.diff(vectorized_dataset.indptr) != 0)==False))
        index = nmslib.init(data_type=data_type, space=space)
        index.addDataPointBatch(vectorized_dataset)
        index.createIndex(print_progress=False)
        print(index)
        return Index(model=model, inverted_index=inverted_index, index=index, aliases=aliases)

    @staticmethod
    def load(path):
        inverted_index = pickle.load(open(os.path.join(path, 'inverted_index.pickle')))
        aliases = pickle.load(open(os.path.join(path, 'aliases.pickle')))
        # model = CompresModel.load(os.path.join(path, 'model'))
        model = pickle.load(open(os.path.join(path, 'model.pickle')))
        index = nmslib.loadIndex(os.path.join(path, 'index'))

        return Index(model=model, inverted_index=inverted_index, index=index, aliases=aliases)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        pickle.dump(self.inverted_index, open(os.path.join(path, 'inverted_index.pickle'), 'wb'))
        pickle.dump(self.aliases, open(os.path.join(path, 'aliases.pickle'), 'wb'))
        pickle.dump(self.model, open(os.path.join(path, 'model.pickle'), 'wb'))
        # self.model.save(os.path.join(path, 'model'))
        self.index.saveIndex(os.path.join(path, 'index'), save_data=True)

    def lookup(self, queries, k=1):
        """
        returns list of (query, [(matching_object, matching_alias, matching_score )])
        """
        vectorized_queries = self.model.predict(queries)
        nn = self.index.knnQueryBatch(vectorized_queries, k=k)
        return [(q, [(self.inverted_index[self.aliases[id_score[0]]], self.aliases[id_score[0]], id_score[1]) for id_score in (zip(*n))],) for q, n in zip(queries, nn)]


