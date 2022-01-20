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
            
            sample = {}
            relation = sent.relations[0] #only one relation per sentence
            sample['docid'] = doc._doc_key
            sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, relation.pair[0].start_doc, relation.pair[0].end_doc, relation.pair[1].start_doc, relation.pair[1].end_doc)
            sample['relation'] = relation.label
            sample['subj_start'] = relation.pair[0].start_doc
            sample['subj_end'] = relation.pair[0].end_doc
            sample['subj_type'] = relation.pair[0].text
            sample['obj_start'] = relation.pair[1].start_doc
            sample['obj_end'] = relation.pair[1].end_doc
            sample['obj_type'] = relation.pair[1].text
            sample['token'] = tokens
            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['nner'] = int(len(sent.ner))

            sent_samples.append(sample)

            # for x in range(len(sent_ner)):
            #     for y in range(len(sent_ner)):
            #         if x == y:
            #             continue
            #         sub = sent_ner[x]
            #         obj = sent_ner[y]
            #         label = gold_rel.get((sub.span, obj.span), 'no_relation')
            #         sample = {}
            #         sample['docid'] = doc._doc_key
            #         sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
            #         sample['relation'] = label
            #         sample['subj_start'] = sub.span.start_sent + sent_start
            #         sample['subj_end'] = sub.span.end_sent + sent_start
            #         sample['subj_type'] = sub.label
            #         sample['obj_start'] = obj.span.start_sent + sent_start
            #         sample['obj_end'] = obj.span.end_sent + sent_start
            #         sample['obj_type'] = obj.label
            #         sample['token'] = tokens
            #         sample['sent_start'] = sent_start
            #         sample['sent_end'] = sent_end
            #         sample['nner'] = int(len(sent.ner))

            #         sent_samples.append(sample)

            # max_sentsample = max(max_sentsample, len(sent_samples))
            # samples += sent_samples
    
    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

    return data, samples, nrel
    
def generate_relation_data_meta_learning(entity_data, use_gold=False, context_window=0, sampling = 'Random'):
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

    ################################### Precompute indexes for entity pair type ###################################
    ent_index_dict = {}
    relation_list  = ['no_relation',
'org:alternate_names',
'org:city_of_branch',
'org:country_of_branch',
'org:dissolved',
'org:founded_by',
'org:founded',
'org:member_of',
'org:members',
'org:number_of_employees/members',
'org:political/religious_affiliation',
'org:shareholders',
'org:stateorprovince_of_branch',
'org:top_members/employees',
'org:website',
'per:age',
'per:cause_of_death',
'per:charges',
'per:children',
'per:cities_of_residence',
'per:city_of_birth',
'per:city_of_death',
'per:countries_of_residence',
'per:country_of_birth',
'per:country_of_death',
'per:date_of_birth',
'per:date_of_death',
'per:employee_of',
'per:identity',
'per:origin',
'per:other_family',
'per:parents',
'per:religion',
'per:schools_attended',
'per:siblings',
'per:spouse',
'per:stateorprovince_of_birth',
'per:stateorprovince_of_death',
'per:stateorprovinces_of_residence',
'per:title']
    for index_data, doc in enumerate(data):
        for i, sent in enumerate(doc):

            nner += len(sent.ner)
            nrel += len(sent.relations)
            if use_gold:
                sent_ner = sent.ner
            else:
                sent_ner = sent.predicted_ner

            relation = sent.relations[0]
            sub_type = ""
            obj_type = ""
            for index in range(len(sent_ner)):
                if(sent_ner[index].span.start_doc == relation.pair[0].start_doc and sent_ner[index].span.end_doc == relation.pair[0].end_doc):
                    sub_type = sent_ner[index].label

                if(sent_ner[index].span.start_doc == relation.pair[1].start_doc and sent_ner[index].span.end_doc == relation.pair[1].end_doc):
                    obj_type = sent_ner[index].label
            
            for rel in relation_list:
                if(rel != relation.label):
                    ent_index_dict.setdefault((sub_type, obj_type, rel), []).append(index_data)

    # print(ent_index_dict)
    ###############################################################################################################


    for doc in data:
        for i, sent in enumerate(doc):
            sent_samples = []
            sent_samples_test= []

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

            sample = {}
            relation = sent.relations[0] #only one relation per sentence
            sample['docid'] = doc._doc_key
            sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, relation.pair[0].start_doc, relation.pair[0].end_doc, relation.pair[1].start_doc, relation.pair[1].end_doc)
            sample['relation'] = relation.label
            #check if first two elements of list are same as start and end 
            sub_type = ""
            obj_type = ""
            for index in range(len(sent_ner)):
                if(sent_ner[index].span.start_doc == relation.pair[0].start_doc and sent_ner[index].span.end_doc == relation.pair[0].end_doc):
                    sub_type = sent_ner[index].label

                if(sent_ner[index].span.start_doc == relation.pair[1].start_doc and sent_ner[index].span.end_doc == relation.pair[1].end_doc):
                    obj_type = sent_ner[index].label
            # print("Subject type = " + sub_type + " Object type = " + obj_type)
            sample['subj_start'] = relation.pair[0].start_doc
            sample['subj_end'] = relation.pair[0].end_doc
            sample['subj_type'] = sub_type # relation.pair[0].text
            sample['obj_start'] = relation.pair[1].start_doc
            sample['obj_end'] = relation.pair[1].end_doc
            sample['obj_type'] = obj_type # relation.pair[1].text
            sample['token'] = tokens
            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['nner'] = int(len(sent.ner))
            sent_samples.append(sample)

            #################################### use the dictionary constructed to get the meta-test samples. ###################################
            index_list = ent_index_dict[(sub_type, obj_type, relation.label)]
            index_test = index_list[random.randint(0,len(index_list)-1)]
            doc_test = data[index_test]
            for i_test, sent_test in enumerate(doc_test):
                relation_test = sent_test.relations[0]
                sent_start_test = 0
                sent_end_test = len(sent_test.text)
                tokens_test = sent_test.text

                if context_window > 0:
                    add_left_test = (context_window-len(sent_test.text)) // 2
                    add_right_test = (context_window-len(sent_test.text)) - add_left_test

                    j_test = i_test - 1
                    while j_test >= 0 and add_left_test > 0:
                        context_to_add_test = doc_test[j_test].text[-add_left_test:]
                        tokens_test = context_to_add_test + tokens_test
                        add_left_test -= len(context_to_add_test)
                        sent_start_test += len(context_to_add_test)
                        sent_end_test += len(context_to_add_test)
                        j_test -= 1

                    j_test = i_test + 1
                    while j_test < len(doc_test) and add_right_test > 0:
                        context_to_add_test = doc_test[j_test].text[:add_right_test]
                        tokens_test = tokens_test + context_to_add_test
                        add_right_test -= len(context_to_add_test)
                        j_test += 1

                sample_test = {}
                sample_test['docid'] = doc_test._doc_key
                sample_test['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc_test._doc_key, sent_test.sentence_ix, relation_test.pair[0].start_doc, relation_test.pair[0].end_doc, relation_test.pair[1].start_doc, relation_test.pair[1].end_doc)
                sample_test['relation'] = relation_test.label
                sample_test['subj_start'] = relation_test.pair[0].start_doc
                sample_test['subj_end'] = relation_test.pair[0].end_doc
                sample_test['subj_type'] = sub_type # relation.pair[0].text
                sample_test['obj_start'] = relation_test.pair[1].start_doc
                sample_test['obj_end'] = relation_test.pair[1].end_doc
                sample_test['obj_type'] = obj_type # relation.pair[1].text
                sample_test['token'] = tokens_test
                sample_test['sent_start'] = sent_start_test
                sample_test['sent_end'] = sent_end_test
                sample_test['nner'] = int(len(sent_test.ner))
                sent_samples_test.append(sample_test)
            #####################################################################################################################################


            #################################### iterate over all samples in training to get the meta-test sample ###################################
            #from all training examples, select one example such that entity type pair matches with the meta-train example but relation is different
            # for doc_test in data:
            #     for i_test, sent_test in enumerate(doc_test):
            #         if use_gold:
            #             sent_ner_test = sent_test.ner
            #         else:
            #             sent_ner_test = sent_test.predicted_ner
                    
            #         #get subject and object type of the current example.
            #         relation_test = sent_test.relations[0] #only one relation per sentence
            #         sub_type_test = ""
            #         obj_type_test = ""
            #         for index in range(len(sent_ner_test)):
            #             if(sent_ner_test[index].span.start_doc == relation_test.pair[0].start_doc and sent_ner_test[index].span.end_doc == relation_test.pair[0].end_doc):
            #                 sub_type_test = sent_ner_test[index].label

            #             if(sent_ner_test[index].span.start_doc == relation_test.pair[1].start_doc and sent_ner_test[index].span.end_doc == relation_test.pair[1].end_doc):
            #                 obj_type_test = sent_ner_test[index].label
                    
            #         sent_start_test = 0
            #         sent_end_test = len(sent_test.text)
            #         tokens_test = sent_test.text

            #         if context_window > 0:
            #             add_left_test = (context_window-len(sent_test.text)) // 2
            #             add_right_test = (context_window-len(sent_test.text)) - add_left_test

            #             j_test = i_test - 1
            #             while j_test >= 0 and add_left_test > 0:
            #                 context_to_add_test = doc_test[j_test].text[-add_left_test:]
            #                 tokens_test = context_to_add_test + tokens_test
            #                 add_left_test -= len(context_to_add_test)
            #                 sent_start_test += len(context_to_add_test)
            #                 sent_end_test += len(context_to_add_test)
            #                 j_test -= 1

            #             j_test = i_test + 1
            #             while j_test < len(doc_test) and add_right_test > 0:
            #                 context_to_add_test = doc_test[j_test].text[:add_right_test]
            #                 tokens_test = tokens_test + context_to_add_test
            #                 add_right_test -= len(context_to_add_test)
            #                 j_test += 1
                        
            #         if(sub_type == sub_type_test and obj_type == obj_type_test and relation.label != relation_test.label):
            #             sample_test = {}
            #             sample_test['docid'] = doc_test._doc_key
            #             sample_test['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc_test._doc_key, sent_test.sentence_ix, relation_test.pair[0].start_doc, relation_test.pair[0].end_doc, relation_test.pair[1].start_doc, relation_test.pair[1].end_doc)
            #             sample_test['relation'] = relation_test.label
            #             sample_test['subj_start'] = relation_test.pair[0].start_doc
            #             sample_test['subj_end'] = relation_test.pair[0].end_doc
            #             sample_test['subj_type'] = sub_type_test # relation.pair[0].text
            #             sample_test['obj_start'] = relation_test.pair[1].start_doc
            #             sample_test['obj_end'] = relation_test.pair[1].end_doc
            #             sample_test['obj_type'] = obj_type_test # relation.pair[1].text
            #             sample_test['token'] = tokens_test
            #             sample_test['sent_start'] = sent_start_test
            #             sample_test['sent_end'] = sent_end_test
            #             sample_test['nner'] = int(len(sent_test.ner))
            #             sent_samples_test.append(sample_test)
            #             break # currently considering only example per meta-train example. Later we can rank these meta-test examples and select most confusing one from the list
            #####################################################################################################################################

            
            # for x in range(len(sent_ner)):
            #     for y in range(len(sent_ner)):
            #         if x == y:
            #             continue
            #         sub = sent_ner[x]
            #         obj = sent_ner[y]
            #         label = gold_rel.get((sub.span, obj.span), 'no_relation')
            #         #Train Sample
            #         sample = {}
            #         sample['docid'] = doc._doc_key
            #         sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
            #         sample['relation'] = label
            #         sample['subj_start'] = sub.span.start_sent + sent_start
            #         sample['subj_end'] = sub.span.end_sent + sent_start
            #         sample['subj_type'] = sub.label
            #         sample['obj_start'] = obj.span.start_sent + sent_start
            #         sample['obj_end'] = obj.span.end_sent + sent_start
            #         sample['obj_type'] = obj.label
            #         sample['token'] = tokens
            #         sample['sent_start'] = sent_start
            #         sample['sent_end'] = sent_end
            #         sample['nner'] = int(len(sent.ner))
            #         sent_samples.append(sample)


            #         #Test Sample
            #         if(sampling == 'Random'):
            #             #Sampling strategy : Random
            #             x_test = random.randrange(len(sent_ner))
            #             y_test = random.randrange(len(sent_ner))
            #             while(x_test != x and y_test != y and x_test != y_test):
            #                 x_test = random.randrange(len(sent_ner))
            #                 y_test = random.randrange(len(sent_ner))
                        
            #         #Sampling strategy : Same entity type
            #         elif(sampling == 'NER'):
            #             flag = False
            #             #intialize with random elements
            #             x_test = random.randrange(len(sent_ner))
            #             y_test = random.randrange(len(sent_ner))
            #             for x_rand in range(len(sent_ner)):
            #                 for y_rand in range(len(sent_ner)):
            #                     if(not flag and sent_ner[x_rand].label == sub.label and sent_ner[y_rand].label == obj.label):
            #                         x_test = x_rand
            #                         y_test = y_rand
            #                         flag = True

            #         sub_test = sent_ner[x_test]
            #         obj_test = sent_ner[y_test]
            #         label_test = gold_rel.get((sub_test.span, obj_test.span), 'no_relation')
            #         sample_test = {}
            #         sample_test['docid'] = doc._doc_key
            #         sample_test['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub_test.span.start_doc, sub_test.span.end_doc, obj_test.span.start_doc, obj_test.span.end_doc)
            #         sample_test['relation'] = label_test
            #         sample_test['subj_start'] = sub_test.span.start_sent + sent_start
            #         sample_test['subj_end'] = sub_test.span.end_sent + sent_start
            #         sample_test['subj_type'] = sub_test.label
            #         sample_test['obj_start'] = obj_test.span.start_sent + sent_start
            #         sample_test['obj_end'] = obj_test.span.end_sent + sent_start
            #         sample_test['obj_type'] = obj_test.label
            #         sample_test['token'] = tokens
            #         sample_test['sent_start'] = sent_start
            #         sample_test['sent_end'] = sent_end
            #         sample_test['nner'] = int(len(sent.ner))
            #         sent_samples_test.append(sample_test)

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples_meta_train += sent_samples

            max_sentsample_test = max(max_sentsample_test, len(sent_samples_test))
            samples_meta_test += sent_samples_test

    tot_train = len(samples_meta_train)
    tot_test = len(samples_meta_test)

    #select a subset of the training samples to see what is the issue with the learning
    # samples_meta_train = samples_meta_train[:5000]
    # samples_meta_test = samples_meta_test[:5000]

    # samples_meta_test = samples_meta_train

    logger.info('#Train-samples: %d, #Test-samples: %d, max #Train-sent.samples: %d, max #Test-sent.samples: %d '%(tot_train, tot_test, max_sentsample, max_sentsample_test))
    #swap meta-train with meta-test
    return data, samples_meta_test, samples_meta_train, nrel
    # return data, samples_meta_train, samples_meta_test, nrel