import json
import logging
import sys
import functools
import random
import os

from shared.data_structures import Dataset

logger = logging.getLogger('root')

def decode_sample_id(sample_id):
    doc_sent = sample_id.split('::')[0]
    pair = sample_id.split('::')[1]
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))

    return doc_sent, sub, obj

def generate_relation_data(entity_data, use_gold=False, context_window=0, is_maml=False):
    """
    Prepare data for the relation model
    If training: set use_gold = True
    """
    logger.info('Generate relation data from %s'%(entity_data))
    data = Dataset(entity_data)

    nner, nrel = 0, 0
    max_sentsample = 0
    samples = []
    for doc in data:
        for i, sent in enumerate(doc):
            sent_samples = []

            nner += len(sent.ner)
            nrel += len(sent.relations)
            if use_gold:
                sent_ner = sent.ner
            else:
                sent_ner = sent.predicted_ner
            
            gold_ner = {}
            for ner in sent.ner:
                gold_ner[ner.span] = ner.label
            
            gold_rel = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = rel.label
            
            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text

            if context_window > 0:
                add_left = (context_window-len(sent.text)) // 2
                add_right = (context_window-len(sent.text)) - add_left

                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    tokens = context_to_add + tokens
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    tokens = tokens + context_to_add
                    add_right -= len(context_to_add)
                    j += 1
            
            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label = gold_rel.get((sub.span, obj.span), 'no_relation')
                    sample = {}
                    sample['docid'] = doc._doc_key
                    sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
                    sample['relation'] = label
                    sample['subj_start'] = sub.span.start_sent + sent_start
                    sample['subj_end'] = sub.span.end_sent + sent_start
                    sample['subj_type'] = sub.label
                    sample['obj_start'] = obj.span.start_sent + sent_start
                    sample['obj_end'] = obj.span.end_sent + sent_start
                    sample['obj_type'] = obj.label
                    sample['token'] = tokens
                    sample['sent_start'] = sent_start
                    sample['sent_end'] = sent_end
                    sample['nner'] = int(len(sent.ner))

                    sent_samples.append(sample)

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples
    
    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

    return data, samples, nrel
    
def generate_relation_data_meta_learning(entity_data, use_gold=False, context_window=0):
    """
    Prepare data for the relation model
    If training: set use_gold = True
    """
    logger.info('Generate relation data from %s'%(entity_data))
    data = Dataset(entity_data)

    nner, nrel = 0, 0
    max_sentsample = 0
    max_sentsample_test = 0
    samples_meta_train = []
    samples_meta_test = []
    for doc in data:
        for i, sent in enumerate(doc):
            sent_samples = []
            sent_samples_test = []

            nner += len(sent.ner)
            nrel += len(sent.relations)
            if use_gold:
                sent_ner = sent.ner
            else:
                sent_ner = sent.predicted_ner
            
            gold_ner = {}
            for ner in sent.ner:
                gold_ner[ner.span] = ner.label
            
            gold_rel = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = rel.label
            
            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text

            if context_window > 0:
                add_left = (context_window-len(sent.text)) // 2
                add_right = (context_window-len(sent.text)) - add_left

                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    tokens = context_to_add + tokens
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    tokens = tokens + context_to_add
                    add_right -= len(context_to_add)
                    j += 1
            
            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label = gold_rel.get((sub.span, obj.span), 'no_relation')
                    #Train Sample
                    sample = {}
                    sample['docid'] = doc._doc_key
                    sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
                    sample['relation'] = label
                    sample['subj_start'] = sub.span.start_sent + sent_start
                    sample['subj_end'] = sub.span.end_sent + sent_start
                    sample['subj_type'] = sub.label
                    sample['obj_start'] = obj.span.start_sent + sent_start
                    sample['obj_end'] = obj.span.end_sent + sent_start
                    sample['obj_type'] = obj.label
                    sample['token'] = tokens
                    sample['sent_start'] = sent_start
                    sample['sent_end'] = sent_end
                    sample['nner'] = int(len(sent.ner))
                    sent_samples.append(sample)

                    #Test Sample
                    
                    x_rand = random.randrange(len(sent_ner))
                    y_rand = random.randrange(len(sent_ner))
                    while(x_rand != x and y_rand != y and x_rand != y_rand):
                        x_rand = random.randrange(len(sent_ner))
                        y_rand = random.randrange(len(sent_ner))
                    
                    sub_test = sent_ner[x_rand]
                    obj_test = sent_ner[y_rand]
                    label_test = gold_rel.get((sub_test.span, obj_test.span), 'no_relation')
                    #Train Sample
                    sample_test = {}
                    sample_test['docid'] = doc._doc_key
                    sample_test['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub_test.span.start_doc, sub_test.span.end_doc, obj_test.span.start_doc, obj_test.span.end_doc)
                    sample_test['relation'] = label_test
                    sample_test['subj_start'] = sub_test.span.start_sent + sent_start
                    sample_test['subj_end'] = sub_test.span.end_sent + sent_start
                    sample_test['subj_type'] = sub_test.label
                    sample_test['obj_start'] = obj_test.span.start_sent + sent_start
                    sample_test['obj_end'] = obj_test.span.end_sent + sent_start
                    sample_test['obj_type'] = obj_test.label
                    sample_test['token'] = tokens
                    sample_test['sent_start'] = sent_start
                    sample_test['sent_end'] = sent_end
                    sample_test['nner'] = int(len(sent.ner))
                    sent_samples_test.append(sample_test)


            max_sentsample = max(max_sentsample, len(sent_samples))
            samples_meta_train += sent_samples

            max_sentsample_test = max(max_sentsample_test, len(sent_samples_test))
            samples_meta_test += sent_samples_test

    tot_train = len(samples_meta_train)
    tot_test = len(samples_meta_test)

    logger.info('#Train-samples: %d, #Test-samples: %d, max #Train-sent.samples: %d, max #Test-sent.samples: %d '%(tot_train, tot_test, max_sentsample, max_sentsample_test))

    return data, samples_meta_train, samples_meta_test, nrel