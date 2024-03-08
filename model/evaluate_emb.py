import pandas
import fsspec
import json
import pickle
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from model.index_emb import Index


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def second_largest(numbers):
    m1 = m2 = m3 = float('-1')
    m1_name = m2_name = m3_name = ''
    for alias, name, x in numbers:
        if x >= m1:
            m1, m1_name, m2, m2_name, m3, m3_name = x, name, m1, m1_name, m2, m2_name
        elif x > m2:
            m2, m2_name, m3, m3_name = x, name, m2, m2_name
        elif x > m3:
            m3, m3_name = x, name
    return m1, m1_name, m2, m2_name,  m3, m3_name

def best_smallest(numbers):
    m1 = m2 = m3 = float('inf')
    m1_name = m2_name = m3_name = ''
    for alias, name, x in numbers:
        if x <= m1:
            m1, m1_name, m2, m2_name, m3, m3_name = x, name, m1, m1_name, m2, m2_name
        elif x < m2:
            m2, m2_name, m3, m3_name = x, name, m2, m2_name
        elif x <m3:
            m3, m3_name = x, name
    return m1, m1_name, m2, m2_name, m3, m3_name


def second_smallest(numbers):
    m1 = m2 = float('inf')
    m1_name = m2_name = ''
    for alias, name, x in numbers:
        if x <= m1:
            m1, m1_name, m2, m2_name = x, name, m1, m1_name
        elif x < m2:
            m2, m2_name = x, name
    return m1, m1_name, m2, m2_name

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


def evaluate(session_dir, model, company_data, stats_dict={}, iteration = -1):
    """ evaluate a model and generate report"""
    if '-sample-' in company_data:
        fin = open(company_data, 'rb')
        data = pickle.load(fin)
    else:
        with fsspec.open(company_data) as company_data_fh:
            data = json.load(company_data_fh)
    index = Index.build_simple_index(model, [x['canonical_company_name'] for x in data['company_list']], use_aliases=False)

    all_companies = pandas.DataFrame(
        [(x['canonical_company_name'], y) for x in data['company_list'] for y in x['synonyms']],
        columns=['name', 'alias']).drop_duplicates().reset_index(drop=True)
    # with Timer('predict aliases'):
    vectorized_aliases = model.predict(all_companies.alias.values)
    # with Timer('predict names'):
    vectorized_names = model.predict(all_companies.name.values)

##### save vectors
    import numpy as np

    # import pickle
    # rel = list(zip(list(all_companies.name.values), list(all_companies.alias.values)))
    # fout = open('/Users/ranziv/Downloads/rel.pickle', 'wb')
    # pickle.dump(rel, fout, protocol=pickle.HIGHEST_PROTOCOL)
    # fout.close()

    #save vectors to file
    # dict_aliases = {}
    # for i in range(0,len(all_companies.alias.values)):
    #     dict_aliases[all_companies.alias.values[i]] = vectorized_aliases[i]
    # # for A, B in np.c_[vectorized_aliases, all_companies.alias.values]:
    # #     dict_aliases[A] = B
    # np.save('/Users/ranziv/Downloads/aliases-data', dict_aliases, allow_pickle=True)
    #
    # dict_names = {}
    # for i in range(0, len(all_companies.name.values)):
    #     dict_names[all_companies.name.values[i]] = vectorized_names[i]
    # # for A, B in np.c_[vectorized_aliases, all_companies.alias.values]:
    # #     dict_aliases[A] = B
    # np.save('/Users/ranziv/Downloads/names-data', dict_names, allow_pickle=True)
    #
    # import pickle
    # fout = open('/Users/ranziv/Downloads/alises.pickle', 'wb')
    # pickle.dump(dict_aliases, fout, protocol=pickle.HIGHEST_PROTOCOL)
    # fout.close()
    #
    # import pickle
    # fout = open('/Users/ranziv/Downloads/names.pickle', 'wb')
    # pickle.dump(dict_names, fout, protocol=pickle.HIGHEST_PROTOCOL)
    # fout.close()
    #
    # fin = open('/Users/ranziv/Downloads/names.pickle', 'rb')
    # b = pickle.load(fin)
    #
    #
    # dict_names = {}
    # tmp_vec_names = np.c_[zip(vectorized_names, all_companies.name.values)]
    # unique_keys, indices = np.unique(tmp_vec_names[:, 0], return_index=True)
    # for A, B in tmp_vec_names[indices]:
    #     dict_names[A] = B
    # np.save('/Users/ranziv/Downloads/names-data', dict_aliases, allow_pickle=True)
    #
    # df_subset={}
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(vectorized_aliases)
    # df_subset['tsne-2d-one'] = tsne_results[:, 0]
    # df_subset['tsne-2d-two'] = tsne_results[:, 1]
    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="tsne-2d-one",§ y="tsne-2d-two",
    #     #hue='tsne-2d-two',
    #     palette=sns.color_palette("hls", 10),
    #     data=df_subset,
    #     legend="full",
    #     alpha=0.3
    # )


    all_companies['alias_name_distance'] = paired_cosine_distances(vectorized_names, vectorized_aliases)


    matches = index.lookup(all_companies.alias, k=3)
    for i in range(3):
        all_companies[f'match_{i}'] = [x[1][i][0] for x in matches]
        all_companies[f'distance_{i}'] = [x[1][i][2] for x in matches]


    # 2021-04-25 confusion_matrix tests
    # https: // medium.com / @ dtuk81 / confusion - matrix - visualization - fc31e3f30fea
    from sklearn.metrics import confusion_matrix
    # Get the confusion matrix
    # cf_matrix = confusion_matrix(all_companies.name[0:100],all_companies.match_0[0:100])
    # print(cf_matrix)
    import seaborn as sns
    # sns.heatmap(cf_matrix, annot=True)
    # sns.heatmap(confusion_matrix(all_companies.name[0:10],all_companies.match_0[0:10]), annot=True)
    import numpy as np
    # sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

    # accuracy = (all_companies.match_0 == all_companies.name).mean()
    accuracy_at_1 = (all_companies.match_0 == all_companies.name).mean()
    accuracy_at_2 = ((all_companies.match_0 == all_companies.name) | (all_companies.match_1 == all_companies.name)).mean()
    accuracy_at_3 = ((all_companies.match_0 == all_companies.name) | (all_companies.match_1 == all_companies.name) | (all_companies.match_2 == all_companies.name)).mean()
    # logger.info({'mean score': accuracy})
    # print({'mean score': accuracy})
    print({'Accuracy Match @ 1': accuracy_at_1})
    print({'Accuracy Match @ 2': accuracy_at_2})
    print({'Accuracy Match @ 3': accuracy_at_3})
    # all_companies.to_csv(f'{session_dir}/report.csv')
    stats_dict[iteration]={}
    stats_dict[iteration]['Accuracy Match @ 1']=accuracy_at_1
    stats_dict[iteration]['Accuracy Match @ 2']=accuracy_at_2
    stats_dict[iteration]['Accuracy Match @ 3']=accuracy_at_3

    #random method
    from random import random
    from random import randrange
    # all_companies[f'random_match'] = [all_companies.name.drop_duplicates().take([randrange(len(all_companies.name.drop_duplicates())-1)]).tolist()[0] for j in range(len(all_companies.name))]
    # accuracy_random = (all_companies.random_match == all_companies.name).mean()
    # print({'Accuracy Random Match': accuracy_random})

    matches_0 = []
    matches_dist_0 = []
    matches_1 = []
    matches_dist_1 = []
    matches_2 = []
    matches_dist_2 = []
    for i in range(0,len(all_companies.alias)):
        alias = all_companies.alias[i]
        aliases_random_distances = []
        for name in all_companies.name:
            aliases_random_distances.append((alias, name, random()))
        m1_dist, m1_name, m2_dist, m2_name, m3_dist, m3_name = best_smallest(aliases_random_distances)
        matches_0.append(m1_name)
        matches_dist_0.append(m1_dist)
        matches_1.append(m2_name)
        matches_dist_1.append(m2_dist)
        matches_2.append(m3_name)
        matches_dist_2.append(m3_dist)
    all_companies[f'random_dist_match_0'] = [x for x in matches_0]
    all_companies[f'random_dist_match_0_dist'] = [x for x in matches_dist_0]
    all_companies[f'random_dist_match_1'] = [x for x in matches_1]
    all_companies[f'random_dist_match_1_dist'] = [x for x in matches_dist_1]
    all_companies[f'random_dist_match_2'] = [x for x in matches_2]
    all_companies[f'random_dist_match_2_dist'] = [x for x in matches_dist_2]
    accuracy_random_distance_0 = (all_companies.random_dist_match_0 == all_companies.name).mean()
    print({'Accuracy Random Distance Match @ 1': accuracy_random_distance_0})
    accuracy_random_distance_1 = ((all_companies.random_dist_match_0 == all_companies.name) | (
    all_companies.random_dist_match_1 == all_companies.name)).mean()
    print({'Accuracy Random Distance Match @ 2': accuracy_random_distance_1})
    accuracy_random_distance_2 = ((all_companies.random_dist_match_0 == all_companies.name) | (
    all_companies.random_dist_match_1 == all_companies.name) | (
                                 all_companies.random_dist_match_2 == all_companies.name)).mean()
    print({'Accuracy Random Distance Match @ 3': accuracy_random_distance_2})
    stats_dict[iteration]['Accuracy Random Distance Match @ 1']= accuracy_random_distance_0
    stats_dict[iteration]['Accuracy Random Distance Match @ 2']= accuracy_random_distance_1
    stats_dict[iteration]['Accuracy Random Distance Match @ 3']= accuracy_random_distance_2







    # Edit (Levenshtein) distance method
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from Levenshtein import jaro_winkler
    from Levenshtein import distance

    # fname = np.array(all_companies.name.drop_duplicates()).reshape(-1, 1)
    # dm = pdist(fname, jaro_winkler)
    # dm = squareform(dm)

    transformed_strings = np.array(all_companies.name.drop_duplicates()).reshape(-1, 1)

    # calculate condensed distance matrix by wrapping the Levenshtein distance function
    # aliases=[]
    # names_no_dups = all_companies.name
    matches_0 = []
    matches_dist_0 = []
    matches_1 = []
    matches_dist_1 = []
    matches_2 = []
    matches_dist_2 = []
    for i in range(0,len(all_companies.alias)):
        alias = all_companies.alias[i]
        aliases_edit_distances = []
        # aliases.append(alias)
        for name in all_companies.name:
            # aliases_edit_distances.append(distance(name, alias))
            aliases_edit_distances.append((alias, name, distance(name, alias)))
        # m1_dist, m1_pos, m2_dist, m2_pos = second_smallest(aliases_edit_distances)
        # m1_dist, m1_name, m2_dist, m2_name = second_smallest(aliases_edit_distances)
        m1_dist, m1_name, m2_dist, m2_name, m3_dist, m3_name = best_smallest(aliases_edit_distances)
        # m1_name=names_no_dups[m1_pos]
        # m2_name=names_no_dups[m2_pos]
        matches_0.append(m1_name)
        matches_dist_0.append(m1_dist)
        matches_1.append(m2_name)
        matches_dist_1.append(m2_dist)
        matches_2.append(m3_name)
        matches_dist_2.append(m3_dist)

    # all_companies[f'edit_dist_match_0'] = [x for x in matches]
    # all_companies[f'edit_dist_match_0_dist'] = [x for x in matches_dist]
    # accuracy_edit_distance = (all_companies.edit_dist_match_0 == all_companies.name).mean()
    # print({'Accuracy Edit Distance Match': accuracy_edit_distance})

    all_companies[f'edit_dist_match_0'] = [x for x in matches_0]
    all_companies[f'edit_dist_match_0_dist'] = [x for x in matches_dist_0]
    all_companies[f'edit_dist_match_1'] = [x for x in matches_1]
    all_companies[f'edit_dist_match_1_dist'] = [x for x in matches_dist_1]
    all_companies[f'edit_dist_match_2'] = [x for x in matches_2]
    all_companies[f'edit_dist_match_2_dist'] = [x for x in matches_dist_2]
    accuracy_edit_distance_0 = (all_companies.edit_dist_match_0 == all_companies.name).mean()
    print({'Accuracy Edit Distance Match @ 1': accuracy_edit_distance_0})
    accuracy_edit_distance_1 = ((all_companies.edit_dist_match_0 == all_companies.name) | (
    all_companies.edit_dist_match_1 == all_companies.name)).mean()
    print({'Accuracy Edit Distance Match @ 2': accuracy_edit_distance_1})
    accuracy_edit_distance_2 = ((all_companies.edit_dist_match_0 == all_companies.name) | (
    all_companies.edit_dist_match_1 == all_companies.name) | (
                                 all_companies.edit_dist_match_2 == all_companies.name)).mean()
    print({'Accuracy Edit Distance Match @ 3': accuracy_edit_distance_2})
    stats_dict[iteration]['Accuracy Edit Distance Match @ 1']= accuracy_edit_distance_0
    stats_dict[iteration]['Accuracy Edit Distance Match @ 2']= accuracy_edit_distance_1
    stats_dict[iteration]['Accuracy Edit Distance Match @ 3']= accuracy_edit_distance_2

    #######################################


    # Fuzzy distance method
    from fuzzywuzzy import fuzz

    matches_0 = []
    matches_dist_0 = []
    matches_1 = []
    matches_dist_1 = []
    matches_2 = []
    matches_dist_2 = []
    for i in range(0, len(all_companies.alias)):
        alias = all_companies.alias[i]
        aliases_fuzzy_distances = []
        # aliases.append(alias)
        for name in all_companies.name:
            aliases_fuzzy_distances.append((alias, name, fuzz.partial_ratio(name, alias)))
        m1_dist, m1_name, m2_dist, m2_name, m3_dist, m3_name = second_largest(aliases_fuzzy_distances)
        matches_0.append(m1_name)
        matches_dist_0.append(m1_dist)
        matches_1.append(m2_name)
        matches_dist_1.append(m2_dist)
        matches_2.append(m3_name)
        matches_dist_2.append(m3_dist)

    all_companies[f'fuzzy_dist_match_0'] = [x for x in matches_0]
    all_companies[f'fuzzy_dist_match_0_dist'] = [x for x in matches_dist_0]
    all_companies[f'fuzzy_dist_match_1'] = [x for x in matches_1]
    all_companies[f'fuzzy_dist_match_1_dist'] = [x for x in matches_dist_1]
    all_companies[f'fuzzy_dist_match_2'] = [x for x in matches_2]
    all_companies[f'fuzzy_dist_match_2_dist'] = [x for x in matches_dist_2]
    accuracy_fuzzy_distance_0 = (all_companies.fuzzy_dist_match_0 == all_companies.name).mean()
    print({'Accuracy Fuzzy Distance Match @ 1': accuracy_fuzzy_distance_0})
    accuracy_fuzzy_distance_1 = ((all_companies.fuzzy_dist_match_0 == all_companies.name) | (all_companies.fuzzy_dist_match_1 == all_companies.name)).mean()
    print({'Accuracy Fuzzy Distance Match @ 2': accuracy_fuzzy_distance_1})
    accuracy_fuzzy_distance_2 = ((all_companies.fuzzy_dist_match_0 == all_companies.name) | (all_companies.fuzzy_dist_match_1 == all_companies.name) | (all_companies.fuzzy_dist_match_2 == all_companies.name)).mean()
    print({'Accuracy Fuzzy Distance Match @ 3': accuracy_fuzzy_distance_2})
    stats_dict[iteration]['Accuracy Fuzzy Distance Match @ 1']= accuracy_fuzzy_distance_0
    stats_dict[iteration]['Accuracy Fuzzy Distance Match @ 2']= accuracy_fuzzy_distance_1
    stats_dict[iteration]['Accuracy Fuzzy Distance Match @ 3']= accuracy_fuzzy_distance_2



    ########################################


    all_companies.to_csv(f'{session_dir}/report.csv')
    return accuracy_at_1, stats_dict


#
#
# accuracy = return true at first position (4k) / all guesses (4k?)
# multiclass accuracy
#
# compare with simple
# random
# edit distance
# fuzzy logic
#
# precision@k
#
#
# {confusion matrix} print in the appendix
# https://en.wikipedia.org/wiki/Confusion_matrix
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
#
# https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
# https://scikit-learn.org/stable/modules/model_evaluation.html
#
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html#sklearn.metrics.top_k_accuracy_score
#
#
#
#
# Damerau-Levenshtein Distance
# Jaro-Winkler Distance
# Soundex
# Double Metaphone
#
#
# Results table:
# method
# accuracy
# top k accuracy
#
#
