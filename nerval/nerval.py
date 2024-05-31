#!/usr/bin/env python
# coding: utf-8

# NER --- Entity-level Confusion Matrix and Classification Report

# Labeling schemes supported:
#
# 1. IO:
#     - I- : inside
#     - O- : outside
#     This scheme cannot distinguish between adjacent chunks of the same named entity.
#
# 2. BIO (or IOB):
#     - B- : begin
#     - I- : inside
#     - O- : outside
#
#     - 2a. BIO1 (or IOB1): the B- tag is only used when a token is the beginning of a chunk immediately following another chunk of the same Named Entity
#           Lucy: I-PER
#           going: O
#           San: I-LOC
#           Francisco: I-LOC
#           California: B-LOC
#
#     - 2b. BIO2 (or IOB2): the B- tag is used for each starting token.
#           Lucy: B-PER
#           going: O
#           San: B-LOC
#           Francisco: I-LOC
#           California: B-LOC
#
# 3. IOE:
#     - I- : inside
#     - O- : outside
#     - E- : end
#
#     - 3a. IOE1: the E- tag is only used when a token is the last token of a chunk immediately preceding another chunk of the same Named Entity.
#     - 3b. IOE2: the E- tag is used for each token that is the last token of a chunk.
#
# 4. IOBES:
#     - I- : inside
#     - O- : outside
#     - B- : begin
#     - E- : end
#     - S- : single (single token)
#     Chunks of length >= 2 always start with the B tag and end with the E tag.
#
#  5. BILOU:
#     - B- : begin
#     - I- : inside
#     - L- : last
#     - O- : outside
#     - U- : unigram (single token)
#     Chunks of length >= 2 always start with the B tag and end with the L tag.
#
# 6. BMEWO:
#     - B- : begin
#     - M- : middle
#     - E- : end
#     - W- : unigram (single token)
#     - O- : outside
#     Chunks of length >= 2 always start with the B tag and end with the L tag.


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import sys



pd.set_option('display.max_row', None)              # show all rows of a dataframe
pd.set_option('display.max_column', None)           # show all columns of a dataframe
pd.set_option('display.max_colwidth', None)         # show the full width of columns
pd.options.display.float_format = '{:,.2f}'.format  # comma separators and two decimal points: 4756.7890 => 4,756.79 and 4656 => 4,656.00
np.set_printoptions(precision=2)                    # numpy print 2 decimal points



# FUNCTIONS
# 1. CLASSIFICATION REPORT & CONFUSION MATRIX
# INPUT to the function is y_true which is the dataset of true tags. It is a list of flat lists.
# INPUT to the function is y_pred which is the dataset of true tags. It is a list of flat lists.
# OUTPUT of the function:
#   - cr: classification report (with micro average, macro average and weighted average at the bottom). This is a Pandas DataFrame.
#   - cm: not normalized confusion matrix (normalization is an option of the plot_confusion_matrix() function). This is a numpy 2d-array.
#   - cm_labels: labels that are needed to plot the matrix. This is a list.

# Choose scheme='BIO' for the following schemes (BIO is the default scheme):
#     - IO
#     - BIO1 (IOB1)
#     - BIO2 (IOB2)
#     - IOBES
#     - BILOU
#     - BMEWO
# Choose scheme='IOE' for the following schemes:
#     - IOE1
#     - IOE2
def crm(y_true, y_pred, scheme='BIO'):

    y_true_clean, y_pred_clean = get_clean_entities(y_true, y_pred, scheme=scheme)       # function

    cr = create_classification_report(y_true_clean, y_pred_clean)                        # function
    cm, cm_labels = create_confusion_matrix(y_true_clean, y_pred_clean)                  # function

    return cr, cm, cm_labels



# 1a. Create confusion matrix with sklearn
# INPUT of the function are two flat lists.
#
#               list_true = ['GEOPOL', 'LOC', 'ORG', 'CURRENCY', 'O', 'O', 'O', 'ORG', 'LOC', 'ORG',
#                            'LOC', 'PER', 'ORG', 'LOC', 'LOC']
#               list_pred = ['GEOPOL', 'LOC', 'ORG', 'CURRENCY', 'LOC', 'ORG', 'TITLE', 'EVENT', 'ORG', 'LOC',
#                            'LOC__', 'PER__', 'O', 'O', 'O']
#
# OUTPUT of the function:
#     - matrix: a numpy 2d-array
#     - labels: a list
#
# NOTE:  This function is embedded into the function crm().
def create_confusion_matrix(list_true, list_pred):

    # Requires flat lists as inputs. Therefore if inputs are lists of lists, the block below flattens them.
    if any(isinstance(s, list) for s in list_true):
        list_true = [item for sublist in list_true for item in sublist]
        list_pred = [item for sublist in list_pred for item in sublist]

    labels = sorted(set(list_true)|set(list_pred))

    matrix = confusion_matrix(list_true, list_pred)                                        # sklearn function

    return matrix, labels



# 1b. Classification Report
# INPUT of the function are two flat lists.
#
#               list_true = ['GEOPOL', 'LOC', 'ORG', 'CURRENCY', 'O', 'O', 'O', 'ORG', 'LOC', 'ORG',
#                            'LOC', 'PER', 'ORG', 'LOC', 'LOC']
#               list_pred = ['GEOPOL', 'LOC', 'ORG', 'CURRENCY', 'LOC', 'ORG', 'TITLE', 'EVENT', 'ORG', 'LOC',
#                            'LOC__', 'PER__', 'O', 'O', 'O']
#
# OUTPUT of the function is a Pandas DataFrame.
#
# NOTE: Divisions by zero are replaced with 0
# NOTE: This function is embedded into the function crm().
def create_classification_report(list_true, list_pred):

    df = performance_metrics(list_true, list_pred)                                           # function

    if df.empty:
        classification_report = pd.DataFrame(data=([]),
                                             index=['-'],
                                             columns=['precision', 'recall', 'f1_score', 'true_entities', 'pred_entities'])

        averages_report = pd.DataFrame(data=([]),
                                       index=['micro_avg', 'macro_avg', 'weighted_avg'],
                                       columns=['precision', 'recall', 'f1_score', 'true_entities', 'pred_entities'])

    else:
        # remove the 'O' row if the 'O' tag is present
        df.drop(['O'], inplace=True) if 'O' in df.index else None

        # 1. CREATE CLASSIFICATION REPORT
        # PRECISION - RECALL - F1_SCORE
        df['precision'] = (df['tp'] / df['pred_entities'])
        df['recall'] = (df['tp'] / df['true_entities'])
        df['f1_score'] = ((2 * df['precision'] * df['recall']) / (df['precision'] + df['recall']))

        # Replace NaN values with 0
        df = df.replace(np.nan, 0)

        # CLASSIFICATION REPORT
        classification_report = df[['precision', 'recall', 'f1_score', 'true_entities',
                                    'pred_entities']].sort_values(by=['true_entities'], ascending=False)


        # 2. CREATE AVERAGES REPORT
        # Calculate totals row
        totals = df.sum()

        # Calculate micro averages
        micro_avg_precision = totals['tp'] / totals['pred_entities'] if totals['pred_entities']!=0 else 0
        micro_avg_recall = totals['tp'] / totals['true_entities'] if totals['true_entities']!=0 else 0
        micro_avg_f1 = ((2 * micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)) if (micro_avg_precision + micro_avg_recall)!=0 else 0

        # Calculate macro averages
<<<<<<< HEAD
        macro_avg_precision = totals['precision'] / len(df) if len(df)!=0 else 0
        macro_avg_recall = totals['recall'] / len(df) if len(df)!=0 else 0
        macro_avg_f1 = totals['f1_score'] / len(df) if len(df)!=0 else 0
=======
        macro_avg_precision = totals['precision'] / len(df) if df else 0
        macro_avg_recall = totals['recall'] / len(df) if df else 0
        macro_avg_f1 = totals['f1_score'] / len(df) if df else 0
>>>>>>> bcf33ea (added git to nerval package on local machine)

        # Calculate weighted averages
        weighted_avg_precision = sum((df['precision'] * df['true_entities'])) / totals['true_entities'] if totals['true_entities']!=0 else 0
        weighted_avg_recall = sum((df['recall'] * df['true_entities'])) / totals['true_entities'] if totals['true_entities']!=0 else 0
        weighted_avg_f1 = sum((df['f1_score'] * df['true_entities'])) / totals['true_entities'] if totals['true_entities']!=0 else 0

        # AVERAGES REPORT
        averages_report = pd.DataFrame(data=([
            [micro_avg_precision, micro_avg_recall, micro_avg_f1, totals['true_entities'], totals['pred_entities']],
            [macro_avg_precision, macro_avg_recall, macro_avg_f1, totals['true_entities'], totals['pred_entities']],
            [weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, totals['true_entities'], totals['pred_entities']]
        ]),
            index=['micro_avg', 'macro_avg', 'weighted_avg'],
            columns=['precision', 'recall', 'f1_score', 'true_entities', 'pred_entities'])


        # 3. CONCAT CLASSIFICATION REPORT & AVERAGES REPORT
        classification_report = pd.concat([classification_report, averages_report], axis=0)

    return classification_report



# 1c. Create a pandas dataframe with the counts by entity of the following:
# 1. tp : true positives, i.e. true=pred
# 2. fn : false negatives, i.e. true!=O pred=O
# 3. fpo : false positives where true=O pred!=O
# 4. fpd_true : false positives where true<>pred and the entity in question is the true entity
# 5. fpd_pred : false positives where true<>pred and the entity in question is the pred entity
# 6. tot_fp : total false positives, i.e. fpo + fpd_true + fpd_pred
# 7. true_entities : total true entities
# 8. pred_entities : total predicted entities
#
# INPUT of the function are two flat lists.
#
#               list_true = ['GEOPOL', 'LOC', 'ORG', 'CURRENCY', 'O', 'O', 'O', 'ORG', 'LOC', 'ORG',
#                            'LOC', 'PER', 'ORG', 'LOC', 'LOC']
#               list_pred = ['GEOPOL', 'LOC', 'ORG', 'CURRENCY', 'LOC', 'ORG', 'TITLE', 'EVENT', 'ORG', 'LOC',
#                            'LOC__', 'PER__', 'O', 'O', 'O']
#
# OUTPUT of the function is a Pandas DataFrame.
#
# NOTE: This function is embedded into the function create_classification_report().
def performance_metrics(list_true, list_pred):

    dict_ = {entity: {'tp': 0, 'fn': 0, 'fpo': 0, 'fpd_true': 0, 'fpd_pred': 0, 'tot_fp': 0,
                      'true_entities': 0, 'pred_entities': 0} for entity in set(list_true)|set(list_pred)}


    for (t_entity, p_entity) in zip(list_true, list_pred):
        dict_[t_entity]['true_entities'] += 1
        dict_[p_entity]['pred_entities'] += 1

        # true posititve
        if t_entity == p_entity:
            dict_[t_entity]['tp'] += 1

        # false negative (TRUE <> 'O', PRED = 'O')
        elif (t_entity != 'O') & (p_entity == 'O'):
            dict_[t_entity]['fn'] += 1
            dict_[p_entity]['fn'] += 1

        # false positive (TRUE = 'O', PRED <> 'O')
        elif (t_entity == 'O') & (p_entity != 'O'):
            dict_[p_entity]['fpo'] += 1
            dict_[t_entity]['fpo'] += 1

            dict_[p_entity]['tot_fp'] += 1
            dict_[t_entity]['tot_fp'] += 1

        # false positive (TRUE <> PRED)
        elif (t_entity != p_entity):
            dict_[t_entity]['fpd_true'] += 1
            dict_[p_entity]['fpd_pred'] += 1

            dict_[t_entity]['tot_fp'] += 1

    df = pd.DataFrame(dict_).transpose()

    return df



# 1d. Clean up entity names
# INPUT to the function is y_true which is the dataset of true tags. It is a list of flat lists.
# INPUT to the function is y_pred which is the dataset of true tags. It is a list of flat lists.
# OUTPUT of the function are two flat lists:
#
#               list_true = ['GEOPOL', 'LOC', 'ORG', 'CURRENCY', 'O', 'O', 'O', 'ORG', 'LOC', 'ORG',
#                            'LOC', 'PER', 'ORG', 'LOC', 'LOC']
#               list_pred = ['GEOPOL', 'LOC', 'ORG', 'CURRENCY', 'LOC', 'ORG', 'TITLE', 'EVENT', 'ORG', 'LOC',
#                            'LOC__', 'PER__', 'O', 'O', 'O']
#
# Note: the __ at the end of some entities means that true and pred have the same name but the prediction is somewhat different from the true label.
# Examples:
# 1) true = ['B-ORG', 'I-ORG', 'I-ORG']) and pred = ['B-ORG']
# 2) true = ['B-ORG', 'I-ORG', 'I-ORG']) and pred = ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG']
# 3) true = ['B-ORG', 'I-ORG', 'I-ORG']) and pred = ['B-ORG', 'I-PER']
# 4) true = ['B-ORG', 'I-ORG', 'I-ORG']) and pred = ['I-ORG', 'I-PER']
#
# NOTE: This function is embedded into the function crm().
def get_clean_entities(y_true, y_pred, scheme='BIO'):

    tp, fn, fpo, fpd = get_pos_neg(y_true, y_pred, scheme=scheme)                            # function

    # TRUE POSITIVE (true = pred)
    t_tp = []
    p_tp = []

    # FALSE POSITIVES (true = 'O')
    # 'o' at the end of the name indicates true label equals 'O'
    t_fpo = []
    p_fpo = []

    # FALSE POSITIVES (true <> pred)
    # 'd' at the end of the name indicates true label different from pred label
    t_fpd = []
    p_fpd = []

    # FALSE NEGATIVES (pred = 'O')
    t_fn = []
    p_fn = []


    # TRUE POSITIVE
    # True and pred are identical, therefore we only need to create one new tag and use it for both entities.
    for entity_pair in tp:
        new_tag = entity_pair[0][0][2:]
        t_tp.append(new_tag)
        p_tp.append(new_tag)

    # FALSE POSITIVE (where true = 'O')
    # True entity always 'O'. We need a new tag only for the pred entity.
    for entity_pair in fpo:
        new_tag = entity_pair[1][0][2:]
        p_fpo.append(new_tag)
        t_fpo.append('O')

    # FALSE POSITIVE (where true <> pred)
    # True and Pred entities are different. We need a new tag for both of them.
    for entity_pair in fpd:
        first_true_tag = entity_pair[0][0][2:]
        first_pred_tag = entity_pair[1][0][2:]

        if first_true_tag == first_pred_tag:
            new_tag_true = entity_pair[0][0][2:]
            new_tag_pred = '{}__'.format(entity_pair[1][0][2:])
        else:
            new_tag_true = entity_pair[0][0][2:]
            new_tag_pred= entity_pair[1][0][2:]

        t_fpd.append(new_tag_true)
        p_fpd.append(new_tag_pred)

    # FALSE NEGATIVE
    # Pred entity always 'O'. We need a new tag only for the true entity.
    for entity_pair in fn:
        new_tag = entity_pair[0][0][2:]
        t_fn.append(new_tag)
        p_fn.append('O')

    list_true = t_tp + t_fpo + t_fpd + t_fn
    list_pred = p_tp + p_fpo + p_fpd + p_fn

    return list_true, list_pred



# 1e. Identify True Positive, False Positives, and False Negatives
# Note: True negatives are not needed because they would mean true=pred='O'
#
# INPUT to the function is y_true which is the dataset of true tags. It is a list of flat lists.
# INPUT to the function is y_pred which is the dataset of true tags. It is a list of flat lists.
# OUTPUT of the function are tp, fn, fpo and fpd. These are all lists of tuples.
#
#              tp = [(['B-GEOPOL'], ['B-GEOPOL']),
#                    (['B-LOC'], ['B-LOC']),
#                    (['B-ORG', 'I-ORG', 'I-ORG'], ['B-ORG', 'I-ORG', 'I-ORG']),
#                    (['B-CURRENCY'], ['B-CURRENCY'])]
#
#              fn = [(['B-ORG', 'I-ORG', 'I-ORG'], 'O'),
#                    (['B-LOC', 'I-LOC', 'I-LOC'], 'O'),
#                    (['B-LOC'], 'O')]
#
#             fpo = [('O', ['B-LOC']),
#                    ('O', ['B-ORG']),
#                    ('O', ['B-TITLE'])]
#
#             fpd = [(['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG'], ['B-EVENT', 'I-EVENT']),
#                    (['B-LOC'], ['B-ORG', 'I-ORG']),
#                    (['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG'], ['B-LOC', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG']),
#                    (['B-LOC', 'I-LOC'], ['B-LOC']),
#                    (['B-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER'], ['B-PER', 'I-PER', 'I-PER'])]
#
# NOTE: This function is embedded into the function get_clean_entities().
def get_pos_neg(y_true, y_pred, scheme='BIO'):

    # TRUE POSITIVE (true = pred)
    tp = []

    # FALSE POSITIVES (true = 'O')
    # 'o' at the end of the name indicates true label equals 'O'
    fpo = []

    # FALSE POSITIVES (true <> pred)
    # 'd' at the end of the name indicates true label different from pred label
    fpd = []

    # FALSE NEGATIVES (pred = 'O')
    fn = []

    # GET ENTITIES AND INDICES
    # The outpus is two lists of tuples
    true_tuples_lst, pred_tuples_lst = get_entities(y_true, y_pred, scheme=scheme)           # function

    for (true_tuple, pred_tuple) in zip(true_tuples_lst, pred_tuples_lst):
        true_entities = true_tuple[0]     # list of lists. Each sublist includes the tags that make up one true entity
        true_indices  = true_tuple[1]     # list of lists. Each sublist includes the corresponding tag indices.
        pred_entities = pred_tuple[0]     # list of lists. Each sublist includes the tags that make up one pred entity
        pred_indices  = pred_tuple[1]     # list of lists. Each sublist includes the corresponding tag indices.

        # GET THE INDEX OF THE FIRST TAG OF EACH ENTITY
        t_start_indices = [lst[0] for lst in true_indices]
        p_start_indices = [lst[0] for lst in pred_indices]

        # FALSE NEGATIVES (pred = 'O')
        for i,(t_entity, t_index) in enumerate(zip(true_entities, true_indices)):
            if t_index[0] not in p_start_indices:
                fn.append((t_entity, 'O'))

        # FALSE POSITIVES (true = 'O')
        for j,(p_entity, p_index) in enumerate(zip(pred_entities, pred_indices)):
            if p_index[0] not in t_start_indices:
                fpo.append(('O', p_entity))

        for i,(t_entity, t_index) in enumerate(zip(true_entities, true_indices)):
            for j,(p_entity, p_index) in enumerate(zip(pred_entities, pred_indices)):

                if (t_index[0] != p_index[0]):
                    continue

                # TRUE POSITIVES (true = pred)
                elif (t_index[0] == p_index[0]) & (t_entity == p_entity):
                    tp.append((t_entity, p_entity))

                # FALSE POSITIVES (true <> pred)
                elif (t_index[0] == p_index[0]) & (t_entity != p_entity):
                    fpd.append((t_entity, p_entity))

    print('True positives: ', len(tp))
    print("False positives (true = 'O'): ", len(fpo))
    print('False positives (true <> pred): ', len(fpd))
    print('ToT False positives: ', len(fpo) + len(fpd))
    print('False negatives: ', len(fn), '\n')

    return tp, fn, fpo, fpd



# 1f. Get entities out of  y_true and y_pred
# INPUT to the function is y_true which is the dataset of true tags. It is a list of flat lists.
# INPUT to the function is y_pred which is the dataset of true tags. It is a list of flat lists.
# OUTPUT of the function is true_tuples and pred_tuples which are lists of tuples:
#
#     pred_tuples_lst = [
#                         ([['B-PER', 'I-PER'], ['B-LOC', 'I-LOC']], [[1, 2], [7, 8]]),
#                         ([['B-DATE'], ['B-ORG', 'I-ORG', 'I-ORG'], ['B-PER']], [[10], [16, 17, 18], [20]])
#                       ]
#
#
#     pred_tuples_lst = [
#                         ([['B-PER', 'I-PER'], ['B-LOC', 'I-LOC']], [[1, 2], [7, 8]]),
#                         ([['B-DATE'], ['B-ORG', 'I-ORG', 'I-ORG'], ['B-PER']], [[10], [16, 17, 18], [20]])
#                       ]
#
# NOTE: This function is embedded into the function get_pos_neg().
def get_entities(y_true, y_pred, scheme='BIO'):

    # check datasets are lists of flat lists (instead of flat lists OR lists of nested lists)
    y_true = check_shape(y_true)                                                                     # function
    y_pred = check_shape(y_pred)                                                                     # function

    true_tuples_lst = dataset_entities(y_true, scheme=scheme)                                        # function
    pred_tuples_lst = dataset_entities(y_pred, scheme=scheme)                                        # function

    count_true_entities, count_true_long_entities = count_entities(true_tuples_lst)                  # function
    count_pred_entities, count_pred_long_entities = count_entities(pred_tuples_lst)                  # function

    print('True Entities: {}'.format(count_true_entities))
    print('Pred Entities: {}'.format(count_pred_entities), '\n')
    print('True Entities with 3 or more tags: {}'.format(count_true_long_entities))
    print('Pred Entities with 3 or more tags: {}'.format(count_pred_long_entities), '\n')

    return true_tuples_lst, pred_tuples_lst



# 1g. Check dataset shape
# NOTE: This function is embedded into the function get_entities().
# checks that dataset is a list of flat lists (instead of a flat list OR a list of nested lists)
def check_shape(dataset):

    if all(isinstance(item, list) for item in dataset) is False:
        dataset = [dataset]                                            # convert a flat list onto a list of flat lists
        print('Dataset was in the wrong shape: flat list.')
        print('It has been converted into the correct shape: list of flat lists.', '\n')
        print('*' * 120)

    elif all(isinstance(item, list) for sublist in dataset for item in sublist) is True:
        dataset = [item for sublist in dataset for item in sublist]    # convert list of nested lists to list of flat lists
        print('Dataset was in the wrong shape: list of nested lists.')
        print('It has been converted into the correct shape: list of flat lists.', '\n')
        print('*' * 120)

    return dataset



# 1h. Identify entities in the entire dataset
# INPUT to the function is dataset which is a list of lists.
#
#             dataset = [
#                          ['O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC'],
#                          ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-DATE', 'O', 'O', 'O', 'O', 'O', 'B-ORG',
#                           'I-ORG', 'I-ORG', 'O', 'B-PER', 'O', 'O', 'O']
#                       ]
#
# OUTPUT of the function is 'tuples' which is a list of tuples.
# Each tuples is made up of two lists of lists. Sublist 1 contains the entities; sublist 2 contains the corresponding indices.
#
#             tuples = [
#                         ([['B-PER', 'I-PER'], ['B-LOC', 'I-LOC']], [[1, 2], [7, 8]]),
#                         ([['B-DATE'], ['B-ORG', 'I-ORG', 'I-ORG'], ['B-PER']], [[10], [16, 17, 18], [20]])
#                      ]
#
# NOTE: This function is embedded into the function get_entities().
def dataset_entities(dataset, scheme='BIO'):

    tuples = []

    for seq in dataset:
        tuples.append(seq_entities(seq, scheme=scheme))             # function

    return tuples



# 1i. Identify entities in one sequence
# INPUT to the function is seq which is a flat list of tags.
#
#               seq = ['O', 'O', 'B-PER', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC']
#
# OUTPUT of the function is 'seq_entities' which is a list of lists. Each sublist contains the tags that make up each entity.
#
#               seq_entities = [['B-PER'], ['B-LOC', 'I-LOC', 'I-LOC']]
#
# OUTPUT of the function is 'indices' which is a list of lists. Each sublist contains the indices of the tags that make up each entity.
#
#               indices = [[2], [6, 7, 8]]
#
# NOTE: This function is embedded into the function dataset_entities().
def seq_entities(seq, scheme='BIO'):

    seq_entities = []
    indices = []

    # The BIO stype includes the following schemes: IO, BIO1, BIO2, IOBES, BILOU, BMEWO, IOB1, IOB2
    if scheme=='BIO':

        for j,tag in enumerate(seq):

            # Skip 'O' tags
            if tag[0] == 'O':
                continue

            # Extract entities where the first tag of the entity starts (correctly) with B-
            elif tag[0] == 'B':
                seq_entities.append([tag])
                indices.append([j])

            # This accounts for mislabeling, e.g. when the first tag of an entity starts with I- (instead of B-)
            # and the entity is at the beginning of the sequence
            elif (tag[0] != 'O') & (tag[0] != 'B') & (j == 0):
                seq_entities.append([tag])
                indices.append([j])

            # This accounts for mislabeling, e.g. when the first tag of an entity starts with I- (instead of B-)
            # and the entitiy is inside the sequence
            elif (tag[0] != 'O') & (tag[0] != 'B') & (j != 0) & (seq[j-1] == 'O'):   # seq[j-1] is the previous tag
                seq_entities.append([tag])
                indices.append([j])

            # This picks up the entity's tags from the second tag on
            elif (tag[0] != 'O') & (tag[0] != 'B') & (j != 0) & (seq[j-1] != 'O'):   # seq[j-1] is the previous tag
                seq_entities[-1].append(tag)
                indices[-1].append(j)


    # The IOE stype includes the following schemes:IOE1 or IOE2
    elif scheme=='IOE':

        for j,tag in enumerate(seq):

            # Skip 'O' tags
            if tag[0] == 'O':
                continue

            elif (seq[j-1][0] == 'O') or (seq[j-1][0] == 'E'):
                seq_entities.append([tag])
                indices.append([j])

            else:
                seq_entities[-1].append(tag)
                indices[-1].append(j)

    else:
        sys.exit("Error! Choose a scheme between 'BIO' or 'IOE'.")

    return seq_entities, indices



# 1j. Count entities:
# - true_entities
# - pred_entities
# - true entities with 3 tags or more
# - pred entities with 3 tags or more
#
# INPUT to the function is a list of tuples:
#
#         tuples_lst = [
#                        ([['B-PER', 'I-PER'], ['B-LOC', 'I-LOC']], [[1, 2], [7, 8]]),
#                        ([['B-DATE'], ['B-ORG', 'I-ORG', 'I-ORG'], ['B-PER']], [[10], [16, 17, 18], [20]])
#                      ]
#
# NOTE: This function is embedded into the function get_entities().
def count_entities(tuples_lst):

    count_entities = 0
    count_long_entities = 0

    for pair in tuples_lst:
        entities_lst = pair[0]

        count_entities += len(entities_lst)

        for entity in entities_lst:
            if len(entity) > 2:
                count_long_entities += 1

    return count_entities, count_long_entities



# 2. PLOT CONFUSION MATRIX
# INPUT to the function are the confusion matrix and the list of labels from the generate_confusion_matrix() function.

# cmap_options = ['Accent','Blues','BrBG','BuGn','BuPu','CMRmap','Dark2','GnBu','Greens','Greys','OrRd','Oranges',
#                 'PRGn','Paired','Pastel1','Pastel2','PiYG','PuBu','PuBuGn','PuOr','PuRd','Purples','RdBu','RdGy',
#                 'RdPu','RdYlBu','RdYlGn','Reds','Set1','Set2','Set3','Spectral','Wistia','YlGn','YlGnBu','YlOrBr',
#                 'YlOrRd','afmhot','autumn','binary','bone','brg','bwr','cividis','cool','coolwarm','copper',
#                 'cubehelix','flag','gist_earth','gist_gray','gist_heat','gist_ncar','gist_rainbow','gist_stern',
#                 'gist_yarg','gnuplot','gnuplot2','gray','hot','hsv','inferno','jet','magma','nipy_spectral',
#                 'ocean','pink','plasma','prism','rainbow','seismic','spring','summer','tab10','tab20','tab20b',
#                 'tab20c','terrain','turbo','twilight','twilight_shifted','viridis','winter']

def plot_confusion_matrix(cm, cm_labels, show=True, save=False, img_path=None, normalize=None,
                          figsize=(35,35), SMALL_SIZE=14, MEDIUM_SIZE=16, BIGGER_SIZE=20, decimal_places=2,
                          cmap='OrRd', xticks_rotation='vertical',title=''):

<<<<<<< HEAD
    if len(cm) == 0:
=======
    if not cm:
>>>>>>> bcf33ea (added git to nerval package on local machine)
        print('Confusion matrix is empty')

    else:

        # Normalise confusion matrix
        cm = normalize_confusion_matrix(cm, normalize=normalize)                                   # function

        # Round confusion_matrix values to n decimal places
        cm = np.around(cm, decimal_places)

        # Set the plot size
        fig, ax = plt.subplots(figsize=figsize)

        # Set font sizes
        SMALL_SIZE = SMALL_SIZE
        MEDIUM_SIZE = MEDIUM_SIZE
        BIGGER_SIZE = BIGGER_SIZE

        plt.rc('font', size=SMALL_SIZE)             # font size of the values in the matrix
        plt.rc('axes', titlesize=BIGGER_SIZE)       # font size of the axes title ('True label' and 'Predicted label')
        plt.rc('axes', labelsize=MEDIUM_SIZE)       # font size of the x and y labels ('B-PER', 'I-EVENT', ...)
        plt.rc('xtick', labelsize=SMALL_SIZE)       # font size of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)       # font size of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)      # legend font size
        plt.rc('figure', titlesize=BIGGER_SIZE)     # font size of the figure title

        # Create the plot
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)           # sklearn function
        cm_display.plot(include_values=True, cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)
        plt.title(title)

        # Save the plot in current dir
<<<<<<< HEAD
        if save == True and img_path == None:
            plt.savefig("confusion_matrix.jpg")

        # Save the plot in dir chosen by user
        if save == True and img_path != None:
=======
        if save == True and img_path is None:
            plt.savefig("confusion_matrix.jpg")

        # Save the plot in dir chosen by user
        if save == True and img_path is not None:
>>>>>>> bcf33ea (added git to nerval package on local machine)
            plt.savefig(img_path)

        # Show the plot
        if show == True:
            plt.show()
        else:
            plt.close()



# 2a. Normalise confusion matrix
# Source: https://github.com/scikit-learn/scikit-learn/blob/582fa30a3/sklearn/metrics/_classification.py#L222

# INPUT of the function is a confusion matrix which is a numpy 2d-array.
# OUTPUT of the function is a confusion matrix which is a numpy 2d-array.
#
# NOTE: This function is embedded into the function plot_confusion_matrix().
def normalize_confusion_matrix(cm, normalize=None):                                                 # normalize: None, 'true', 'pred', 'all'

    # To avoid error message when dividing by zero or dividing by NaN
    with np.errstate(all='ignore'):

<<<<<<< HEAD
        if normalize == None:
=======
        if normalize is None:
>>>>>>> bcf33ea (added git to nerval package on local machine)
            cm = cm

        elif normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)

        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)

        elif normalize == 'all':
            cm = cm / cm.sum()

        # nan_to_num() is used when we want to replace NaN with zero and inf with finite numbers in an array.
        # It replaces (positive) infinity with a very large number and negative infinity with a very small (or negative)
        # number.
        cm = np.nan_to_num(cm)

    return cm
