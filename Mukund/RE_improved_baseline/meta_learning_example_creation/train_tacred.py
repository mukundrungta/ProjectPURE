import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from prepro import TACREDProcessor
# from evaluation import get_f1
# from evaluation import accuracy_challenge_tacred
from model import REModel
from torch.cuda.amp import GradScaler
import wandb
import pandas as pd
import pandas as pd
# import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import List
import json
from itertools import accumulate, groupby
from collections import namedtuple
from collections import defaultdict, namedtuple, Counter
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk import ngrams
import spacy
import string


Document = namedtuple("Document", ["id", "tokens", "gt_relation","count_token","sub_start","obj_start","attr_dist","sub_type","obj_type","sub_name","obj_name","sub_end","obj_end"])
def load_tacred(path: str) -> List[Document]:
    dataset_document = {}
    with open(path, "r") as input_f:
        dataset = json.load(input_f)

        documents = {}
        for example in dataset:
            # print(example)
            raw_tokens = example["token"]
            id_ = example["id"]
            gt_relation = example["relation"]
            head_start, head_end = example["subj_start"], example["subj_end"]
            tail_start, tail_end = example["obj_start"], example["obj_end"]
            head_tag = example["subj_type"]
            tail_tag = example["obj_type"]

            subject_name = " ".join([raw_tokens[i] for i in range(head_start, head_end+1)])
            object_name = " ".join([raw_tokens[i] for i in range(tail_start, tail_end+1)])

            
            subject_ner = head_tag
            object_ner = tail_tag
            attr_dist = abs(example["obj_start"] - example["subj_end"]-1) if (example["obj_start"] > example["subj_end"]) else abs(example["subj_start"] - example["obj_end"]-1)
            dataset_document[id_] = Document(id=id_,
                                      tokens=raw_tokens,
                                      gt_relation=gt_relation,
                                      count_token=len(raw_tokens),
                                      sub_start=head_start,
                                      obj_start=tail_start,
                                      sub_end=head_end,
                                      obj_end=tail_end,
                                      attr_dist=attr_dist,
                                      sub_type=head_tag,
                                      obj_type=tail_tag,
                                      sub_name=subject_name,
                                      obj_name=object_name) 
    return dataset_document

def token2sent(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


df = pd.read_csv('/nethome/mrungta8/project/ProjectPURE/Mukund/cartography/dataframe_TACRED_BERT.csv')  
df_hard = df.loc[df['correct.'].isin([0.0, 0.2])]
print("Size of hard = " +str(len(df_hard)))

df_amb = df.loc[df['correct.'].isin([0.2, 0.4, 0.6, 0.8])]
print("Size of Ambiguous = " +str(len(df_amb)))

df_easy = df.loc[df['correct.'].isin([0.8, 1.0])]
print("Size of Easy = " +str(len(df_easy)))

guid_easy = list(df_easy['guid'])
guid_hard = list(df_hard['guid'])
guid_amb = list(df_amb['guid'])

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
ID_TO_LABEL = {}
for key in LABEL_TO_ID.keys():
    ID_TO_LABEL[LABEL_TO_ID[key]] = key

dataset_epoch_prediction = {}
for index, row in df.iterrows():
    guid = row['guid']
    pred_labels = []
    pred = json.loads(row['predicted_label'])
    for epoch, label in enumerate(pred):
        pred_labels.append(ID_TO_LABEL[label])
    dataset_epoch_prediction[guid] = pred_labels

##################################### Select 33% of the high variability examples as meta-test examples TACRED #####################################
with open('/nethome/mrungta8/project/ProjectPURE/Mukund/RE_improved_baseline/dataset/tacred_filtered_variability/train.json', "r") as fh:
    data = json.load(fh)

guid_meta_test_variability = []
for d in tqdm(data):
    guid_meta_test_variability.append(d['id'])
print("Number of examples for meta-test-list (considering only 33 high variability cases) are " +str(len(guid_meta_test_variability)))
#########################################################################################################################################


def perform_preprocessing(dataset_document):
    count_easy = 0
    meta_test_filtered_list = {}
    for guid in tqdm(guid_easy):
        meta_test_filtered_list[guid] = []
        guid_meta_test = guid_meta_test_variability # guid_amb #+ guid_hard
        for test_guid in guid_meta_test:
            if(dataset_document[guid].sub_type == dataset_document[test_guid].sub_type and #subject entity type should be same as meta-train
                dataset_document[guid].obj_type == dataset_document[test_guid].obj_type and #object entity type should be same as meta-train
                dataset_document[guid].gt_relation != dataset_document[test_guid].gt_relation and #second condition : gold label should be different from the meta-train
                dataset_document[guid].gt_relation in dataset_epoch_prediction[test_guid][:2] # first 3 epochs of the Data-Cartography phase, the model should make the same label prediction (at least once) as meta-train
                ):
                meta_test_filtered_list[guid].append(test_guid)
        if(len(meta_test_filtered_list[guid])>0):
            count_easy += 1

    print("Total number of easy examples = " +str(len(guid_easy)))
    print("Easy examples for which meta-test exists = " +str(count_easy))

    return meta_test_filtered_list

def compute_ngram(tokens, nlp, translator, n):
    token_list = []
    for token_idx, token_text in enumerate(tokens):
        token_list.append(token_text)
    sentence = TreebankWordDetokenizer().detokenize(token_list).lower()
    sentence_augmented = nlp(sentence)
    sentence_filtered = " ".join([ent.text for ent in sentence_augmented if not ent.ent_type_])
    # print(" ".join([ent.text for ent in sentence_augmented if not ent.ent_type_]))

    tokens = TreebankWordTokenizer().tokenize(sentence_filtered.translate(translator))
    ngram_list = []
    if(len(tokens) >= n):
        ngram_list = ngrams(tokens, n)
        ngram_list = [ ' '.join(grams) for grams in ngram_list]
    return ngram_list

def train_ngram(args, dataset_document, meta_test_filtered_list, n):
    nlp = spacy.load('en_core_web_sm')
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space

    dict_meta_test_map = {} #final list of mapping  between guid of easy cases to the selected example from amb + hard
    ########### finding the ngram for all meta-test beforehand ###########
    print("Started Computing ngram for the meta-test examples")
    dict_test_guid2ngram = {}
    guid_meta_test = guid_meta_test_variability  #guid_amb #+ guid_hard
    for test_guid in guid_meta_test:
        document = dataset_document[test_guid]
        dict_test_guid2ngram[test_guid] = compute_ngram(document.tokens, nlp, translator, n)
    print("Computed representation for the meta-test examples beforehand")
    #############################################################################################

    for guid in tqdm(guid_easy):
        easy_document = dataset_document[guid]
        meta_train_ngram = compute_ngram(easy_document.tokens, nlp, translator, n)
        distance_list = []
        for meta_test_guid in meta_test_filtered_list[guid]:
            meta_test_ngram = dict_test_guid2ngram[meta_test_guid]
            overlap = list(set(meta_train_ngram).intersection(meta_test_ngram))
            score = 0.0
            if(len(meta_test_ngram) > 0):
                score = len(overlap)*1.0 / len(meta_test_ngram)
            distance_list.append(score)
        if(len(distance_list) > 0):
            distance_list = np.asarray(distance_list)
            index = np.argmax(distance_list)
            sampled_meta_test_guid = meta_test_filtered_list[guid][index]
            dict_meta_test_map[guid] = sampled_meta_test_guid
    
    return dict_meta_test_map


def train(args, model, train_features, meta_test_filtered_list):

    dict_meta_test_map = {} #final list of mapping  between guid of easy cases to the selected example from amb + hard
    
    ########### finding the dict of model_representation for all meta-test beforehand ###########
    print("Started Computing representation for the amb + hard examples")
    dict_test_guid2rep = {}
    guid_meta_test = guid_meta_test_variability  #guid_amb #+ guid_hard
    for test_guid in guid_meta_test:
        test_feature = [train_features[test_guid]]
        test_feature = DataLoader(test_feature, batch_size=1, shuffle=True, collate_fn=collate_fn, drop_last=True)
        batch_test= next(iter(test_feature))
        test_feature_inputs = {'input_ids': batch_test[0].to(args.device),
                    'attention_mask': batch_test[1].to(args.device),
                    'labels': batch_test[2].to(args.device),
                    'ss': batch_test[3].to(args.device),
                    'os': batch_test[4].to(args.device),
                    }

        test_model_output = model(**test_feature_inputs)
        test_model_output = test_model_output.cpu().data.numpy()
        dict_test_guid2rep[test_guid] = test_model_output
    print("Computed representation for the amb + hard examples beforehand")
    #############################################################################################
    for guid in tqdm(guid_easy):
        easy_feature = [train_features[guid]]
        easy_feature = DataLoader(easy_feature, batch_size=1, shuffle=True, collate_fn=collate_fn, drop_last=True)
        batch = next(iter(easy_feature))
        easy_feature_inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'ss': batch[3].to(args.device),
                      'os': batch[4].to(args.device),
                      }
        
        easy_model_output = model(**easy_feature_inputs)
        easy_model_output = easy_model_output.cpu().data.numpy()
        distance_list = []
        for meta_test_guid in meta_test_filtered_list[guid]:
            # meta_test_feature = [train_features[meta_test_guid]]
            # meta_test_feature = DataLoader(meta_test_feature, batch_size=1, shuffle=True, collate_fn=collate_fn, drop_last=True)
            # batch_meta_test= next(iter(meta_test_feature))
            # meta_test_feature_inputs = {'input_ids': batch_meta_test[0].to(args.device),
            #           'attention_mask': batch_meta_test[1].to(args.device),
            #           'labels': batch_meta_test[2].to(args.device),
            #           'ss': batch_meta_test[3].to(args.device),
            #           'os': batch_meta_test[4].to(args.device),
            #           }

            # meta_test_model_output = model(**meta_test_feature_inputs)
            # meta_test_model_output = meta_test_model_output.cpu().data.numpy()
            meta_test_model_output = dict_test_guid2rep[meta_test_guid]
            eucledian_distance = np.linalg.norm(easy_model_output - meta_test_model_output)
            distance_list.append(eucledian_distance)
        
        if(len(distance_list) > 0):
            distance_list = np.asarray(distance_list)
            index = np.argmax(distance_list)
            sampled_meta_test_guid = meta_test_filtered_list[guid][index]
            dict_meta_test_map[guid] = sampled_meta_test_guid
        # else:
        #     dict_meta_test_map[guid] = -1

    return dict_meta_test_map


def getSentenceFirstColumn(sentenceToken, subStart, subEnd, objStart, objEnd, subType, objType, sentenceStart):
    s = ""
    index = 0

    subStart = subStart - sentenceStart
    subEnd = subEnd - sentenceStart
    objStart = objStart - sentenceStart
    objEnd = objEnd - sentenceStart
    sentenceTokenTemp = list(sentenceToken)
    sentenceTokenTemp[subStart] = "<SUBJECT-" + subType + ">" +sentenceTokenTemp[subStart]
    sentenceTokenTemp[subEnd] = sentenceTokenTemp[subEnd] + "</" + subType + "-SUBJECT>"

    sentenceTokenTemp[objStart] = "<OBJECT-" + objType + ">" +sentenceTokenTemp[objStart]
    sentenceTokenTemp[objEnd] = sentenceTokenTemp[objEnd] + "</" + objType + "-OBJECT>"
    s = token2sent(sentenceTokenTemp)
    return s

def create_df(dict_meta_test_map,dataset_document):
    easy_sentences = []
    meta_sentences = []
    easy_relations = []
    meta_relations = []
    for key in dict_meta_test_map.keys():
        value  = dict_meta_test_map[key]

        easy_document = dataset_document[key]
        easy_sent = getSentenceFirstColumn(easy_document.tokens, easy_document.sub_start, easy_document.sub_end, easy_document.obj_start, easy_document.obj_end, easy_document.sub_type, easy_document.obj_type, 0)
        easy_sentences.append(easy_sent)
        easy_relations.append(easy_document.gt_relation)
        
        meta_document = dataset_document[value]
        meta_sent = getSentenceFirstColumn(meta_document.tokens, meta_document.sub_start, meta_document.sub_end, meta_document.obj_start, meta_document.obj_end, meta_document.sub_type, meta_document.obj_type, 0)
        meta_sentences.append(meta_sent)
        meta_relations.append(meta_document.gt_relation)
    
    d = {'Sentence_Easy': easy_sentences, 'GT_Relation_Easy': easy_relations, 'Sentence_Meta' : meta_sentences, 'GT_Relation_Meta': meta_relations}
    df_final = pd.DataFrame(d)
    return df_final


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/tacred", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=42)
    parser.add_argument("--evaluation_steps", type=int, default=2500,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--run_name", type=str, default="tacred")

    args = parser.parse_args()
    wandb.init(project=args.project_name, name=args.run_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # if args.seed > 0:
    #     set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )

    config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    model = REModel(args, config)
    model.to(0)

    # train_file = os.path.join(args.data_dir, "train.json")
    train_file = '/nethome/mrungta8/project/ProjectPURE/Mukund/RE_improved_baseline/dataset/tacred/train.json'
    # dev_file = os.path.join(args.data_dir, "dev.json")
    # test_file = os.path.join(args.data_dir, "test.json")
    # dev_rev_file = os.path.join(args.data_dir, "dev_rev.json")
    # test_rev_file = os.path.join(args.data_dir, "test_rev.json")
    # challenge_test_file = os.path.join(args.data_dir, "challenge_set.json")

    processor = TACREDProcessor(args, tokenizer)
    train_features = processor.read(train_file) #dictionary with id as key as features as value
    # dev_features = processor.read(dev_file)
    # test_features = processor.read(test_file)
    # dev_rev_features = processor.read(dev_rev_file)
    # test_rev_features = processor.read(test_rev_file)
    # challenge_test_features = processor.read(challenge_test_file)

    dataset_document = load_tacred(train_file)
    print("****************************** Starting preprocessing of the dataset ******************************")
    meta_test_filtered_list = perform_preprocessing(dataset_document)
    print("****************************** Preprocessing completed ********************************************")


    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))


    # dict_meta_test_map = train(args, model, train_features, meta_test_filtered_list)
    dict_meta_test_map = train_ngram(args, dataset_document, meta_test_filtered_list, n=1) #compute dictionary of meta-train and meta-test pair using the ngram similarity

    with open('dict_meta_test_map_high_variability_ngram.pkl', 'wb') as f:
        pickle.dump(dict_meta_test_map, f)
    df = create_df(dict_meta_test_map, dataset_document)
    df.to_csv('meta-learning-example_high_variability_ngram.csv', index=False)


if __name__ == "__main__":
    main()
