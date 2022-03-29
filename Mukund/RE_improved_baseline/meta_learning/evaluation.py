import numpy as np
import sys
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from collections import Counter

def get_f1(key, prediction):
    correct_by_relation = ((key == prediction) & (prediction != 0)).astype(np.int32).sum()
    guessed_by_relation = (prediction != 0).astype(np.int32).sum()
    gold_by_relation = (key != 0).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro

NO_RELATION = "no_relation"

def compute_f1(preds, labels):
    """Compute precision, recall and f1 as a row data """

    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != NO_RELATION:
            n_pred += 1
        if label != NO_RELATION:
            n_gold += 1
        if (pred != NO_RELATION) and (label != NO_RELATION) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}

def accuracy_challenge_retacred(key, prediction):
    LABEL_TO_ID = {'no_relation': 0, 'org:founded_by': 1, 'per:identity': 2, 'org:alternate_names': 3, 'per:children': 4, 'per:origin': 5, 'per:countries_of_residence': 6, 'per:employee_of': 7, 'per:title': 8, 'org:city_of_branch': 9, 'per:religion': 10, 'per:age': 11, 'per:date_of_death': 12, 'org:website': 13, 'per:stateorprovinces_of_residence': 14, 'org:top_members/employees': 15, 'org:number_of_employees/members': 16, 'org:members': 17, 'org:country_of_branch': 18, 'per:spouse': 19, 'org:stateorprovince_of_branch': 20, 'org:political/religious_affiliation': 21, 'org:member_of': 22, 'per:siblings': 23, 'per:stateorprovince_of_birth': 24, 'org:dissolved': 25, 'per:other_family': 26, 'org:shareholders': 27, 'per:parents': 28, 'per:charges': 29, 'per:schools_attended': 30, 'per:cause_of_death': 31, 'per:city_of_death': 32, 'per:stateorprovince_of_death': 33, 'org:founded': 34, 'per:country_of_death': 35, 'per:country_of_birth': 36, 'per:date_of_birth': 37, 'per:cities_of_residence': 38, 'per:city_of_birth': 39}
    ID_TO_LABEL = {}
    for k in LABEL_TO_ID.keys():
        ID_TO_LABEL[LABEL_TO_ID[k]] = k
    key_label = []
    pred_label = []
    for k in key:
        key_label.append(ID_TO_LABEL[k])
    for p in prediction:
        pred_label.append(ID_TO_LABEL[p])
    
    with open("dataset/retacred/challenge_set.json", "r") as tacred_test_file:
        gold_data = json.load(tacred_test_file)
    
    pred_y=[]
    # print("Analyze the data allignment")
    # gold_data = gold_data[:1000]
    for index, row in enumerate(gold_data):
        #check if true labels are same
        # print("from code = " + str(key_label[index]) + " From json = " +str(row['gold_relation']))
        if(pred_label[index] == row['id_relation']):
            pred_y.append(row['id_relation'])
        else:
            pred_y.append(NO_RELATION)
    
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for row, pp, tt in zip(gold_data, pred_y, key_label):

        curr_id_relation = row["id_relation"]

        true_positive += 1  if pp == curr_id_relation and tt == curr_id_relation else 0
        false_positive += 1 if pp == curr_id_relation and tt != curr_id_relation else 0
        true_negative += 1  if pp != curr_id_relation and tt != curr_id_relation else 0
        false_negative += 1 if pp != curr_id_relation and tt == curr_id_relation else 0

        # print(row['id_relation'], get_clor_entitis(make_readable_sampl(row)['token']))

    print("ACCURACY:   {:.2%} \n".format(accuracy_score(key_label, pred_y)))

    total_accuracy_score_positive = true_positive / (true_positive + false_negative)
    total_accuracy_score_negative = true_negative / (false_positive + true_negative)

    print("POSITIVE ACCURACY:   {:.2%} \n".format(total_accuracy_score_positive))
    print("NEGATIVE ACCURACY:   {:.2%} \n".format(total_accuracy_score_negative))

    print("-------------------------------------------------------------------\n")

    number_of_examples = len(key_label)

    print("TRUE POSITIVE:   {:.3f} \t\t (NUMBER:   {})\n".format(true_positive / number_of_examples, true_positive))
    print("FALSE POSITIVE:   {:.3f} \t\t (NUMBER:   {})\n".format(false_positive / number_of_examples, false_positive))
    print("TRUE NEGATIVE:   {:.3f} \t\t (NUMBER:   {})\n".format(true_negative / number_of_examples, true_negative))
    print("FALSE NEGATIVE:   {:.3f} \t\t (NUMBER:   {})\n".format(false_negative / number_of_examples, false_negative))

    print("-------------------------------------------------------------------\n")

    f1 = compute_f1(pred_y, key_label)

    print("Precision: {:.2%}\t Recall: {:.2%}\t  F1: {:.2%}\n".format(f1["precision"], f1["recall"], f1["f1"]))

def accuracy_challenge_tacred(key, prediction):
    LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
    ID_TO_LABEL = {}
    for k in LABEL_TO_ID.keys():
        ID_TO_LABEL[LABEL_TO_ID[k]] = k
    key_label = []
    pred_label = []
    for k in key:
        key_label.append(ID_TO_LABEL[k])
    for p in prediction:
        pred_label.append(ID_TO_LABEL[p])
    
    with open("dataset/tacred/challenge_set.json", "r") as tacred_test_file:
        gold_data = json.load(tacred_test_file)
    
    pred_y=[]
    # print("Analyze the data allignment")
    # gold_data = gold_data[:1000]
    for index, row in enumerate(gold_data):
        #check if true labels are same
        # print("from code = " + str(key_label[index]) + " From json = " +str(row['gold_relation']))
        if(pred_label[index] == row['id_relation']):
            pred_y.append(row['id_relation'])
        else:
            pred_y.append(NO_RELATION)
    
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for row, pp, tt in zip(gold_data, pred_y, key_label):

        curr_id_relation = row["id_relation"]

        true_positive += 1  if pp == curr_id_relation and tt == curr_id_relation else 0
        false_positive += 1 if pp == curr_id_relation and tt != curr_id_relation else 0
        true_negative += 1  if pp != curr_id_relation and tt != curr_id_relation else 0
        false_negative += 1 if pp != curr_id_relation and tt == curr_id_relation else 0

        # print(row['id_relation'], get_clor_entitis(make_readable_sampl(row)['token']))

    print("ACCURACY:   {:.2%} \n".format(accuracy_score(key_label, pred_y)))

    total_accuracy_score_positive = true_positive / (true_positive + false_negative)
    total_accuracy_score_negative = true_negative / (false_positive + true_negative)

    print("POSITIVE ACCURACY:   {:.2%} \n".format(total_accuracy_score_positive))
    print("NEGATIVE ACCURACY:   {:.2%} \n".format(total_accuracy_score_negative))

    print("-------------------------------------------------------------------\n")

    number_of_examples = len(key_label)

    print("TRUE POSITIVE:   {:.3f} \t\t (NUMBER:   {})\n".format(true_positive / number_of_examples, true_positive))
    print("FALSE POSITIVE:   {:.3f} \t\t (NUMBER:   {})\n".format(false_positive / number_of_examples, false_positive))
    print("TRUE NEGATIVE:   {:.3f} \t\t (NUMBER:   {})\n".format(true_negative / number_of_examples, true_negative))
    print("FALSE NEGATIVE:   {:.3f} \t\t (NUMBER:   {})\n".format(false_negative / number_of_examples, false_negative))

    print("-------------------------------------------------------------------\n")

    f1 = compute_f1(pred_y, key_label)

    print("Precision: {:.2%}\t Recall: {:.2%}\t  F1: {:.2%}\n".format(f1["precision"], f1["recall"], f1["f1"]))