task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['org:alternate_names',
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
'per:title'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
