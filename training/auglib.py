from iteration_utilities import unique_everseen
import pandas as pd


def extract_iob_group_entities(data, output_format='dict', remove_duplicates=False):
    results = []
    match_group, b, i = [], False, False
    for w in data:
        if w[1].startswith('B') and not i and not b: 
            match_group.append(w)
            b = True
        elif w[1].startswith('I') and b and not i:
            match_group.append(w)
            i = True
            b = False
        elif w[1].startswith('I') and not b:
            match_group.append(w)
            i = True
        elif w[1].startswith('B') and i and not b: 
            results.append(match_group)
            match_group, b, i = [], True, False
            match_group.append(w)
        elif w[1].startswith('B') and not i and b: 
            results.append(match_group)
            match_group = []
            match_group.append(w)            
        else:
            print('----------------', w)

    if remove_duplicates:
        results = list(unique_everseen(results, key=list))

    if output_format == 'dict':
        results = [dict(x) for x in results]

    return results


def detect_iob_tag_position(iob_sent): 
    results = []
    match_group, b, i = [], False, False
    for id, w in enumerate(iob_sent):
        v = list(w)
        if w[1].startswith('B') and not i and not b: 
            v.append(id)
            match_group.append(v)
            b = True
        elif w[1].startswith('I') and b and not i:
            v.append(id)
            match_group.append(v)
            i = True
            b = False
        elif w[1].startswith('I') and not b:
            v.append(id)
            match_group.append(v)
            i = True
        elif w[1].startswith('B') and i and not b: 
            results.append(match_group)
            match_group, b, i = [], True, False
            v.append(id)
            match_group.append(v)
        elif w[1].startswith('B') and not i and b: 
            results.append(match_group)
            match_group = []
            v.append(id)
            match_group.append(v)
        if len(iob_sent)==id+1 and len(match_group) == 1 and len(results) == 0:
            results.append(match_group)
        elif len(iob_sent)==id+1: results.append(match_group)

    return results


def detect_iob_type(data):
    out = [x[1] for x in data][0].split('-')[1]
    return out


def check_ner_type(ner_data):
    if 'I-DATE' in ner_data or 'B-LOC' in ner_data or 'B-PER' in ner_data \
        or 'I-PER' in ner_data or 'B-DATE' in ner_data or 'B-ORG' in ner_data \
        or 'I-ORG' in ner_data or 'I-LOC' in ner_data:
        return True
    else: return False

# function that read IOB file and build data structure for train, test and dev
def read_format_iob_data(filename):
    sents_id, words, iob_tag = [], [], []
    all_extracted_data, only_ner_data, o_ner_data = [], [], []
    with open(filename, encoding='utf-8') as iob:
        sentence, id_sent, tags = [], 1, []
        for line in iob:
            if len(line) > 1:
                word, tag = line.strip().split(' ')
                sentence.append((word, tag))
                sents_id.append(id_sent)
                words.append(word)
                iob_tag.append(tag)
                tags.append(tag)
            else:
                if sentence[-1] != '.': 
                    sentence.append(('.', 'O'))
                    words.append('.')
                    iob_tag.append('O')
                sents_id.append(id_sent)
                all_extracted_data.append(sentence)
                if check_ner_type(tags): only_ner_data.append(sentence)
                else: o_ner_data.append(sentence)
                sentence = []
                id_sent += 1
                tags = []
    dataframe = {"sentence_id": sents_id, "word": words, "iob_tag": iob_tag}
    pd_iob_data = pd.DataFrame.from_dict(dataframe)
    return all_extracted_data, pd_iob_data, only_ner_data, o_ner_data


def augment_sentence(sentence, list_ent_aug=None):
    # pour la phrase en entrée, générer n phrases supplémentaires à partir 
    # des entités fournies
    word_positions = detect_iob_tag_position(sentence)
    results_aug = []
    #print('\n=> ', " ".join([i[0] for i in sentence]), '---', word_positions, end='\n\n')
    for word in word_positions:
        tag = detect_iob_type(word)
        sent_aug = []
        for entity in list_ent_aug[tag]:
            pos = [x[2] for x in word]
            for i, x in enumerate(sentence):
                if i not in pos: sent_aug.append(x)
                else: 
                    #print(x)
                    if entity[0] not in sent_aug and word not in entity: 
                        for e in entity: sent_aug.append(e)

            if sent_aug not in results_aug: 
                results_aug.append(sent_aug)
            sent_aug = []
                
            #print('---', word, '---', pos, ' --- ',entity)
            #print('\n---> ', results_aug)
    return results_aug


def augment_ner_iob_data(train_data):
    """Position to position augmentation: generate alternate sentence base on 
    entities position of the sentence and all others entities groups

    Args:
        train_data (list): list of input sentences in iob format

    Returns:
        list: list of generate alternate sentences
    """
    org_list  = [x for sent in train_data for x in sent if x[1] in ['B-ORG','I-ORG']]
    date_list = [x for sent in train_data for x in sent if x[1] in ['B-DATE','I-DATE']]
    loc_list  = [x for sent in train_data for x in sent if x[1] in ['B-LOC','I-LOC']]
    per_list  = [x for sent in train_data for x in sent if x[1] in ['B-PER','I-PER']]
    
    org_list_group  = extract_iob_group_entities(org_list, output_format='list', remove_duplicates=True)
    date_list_group = extract_iob_group_entities(date_list, output_format='list', remove_duplicates=True)
    loc_list_group  = extract_iob_group_entities(loc_list, output_format='list', remove_duplicates=True)
    per_list_group  = extract_iob_group_entities(per_list, output_format='list', remove_duplicates=True)

    ents_groups = {'ORG': org_list_group, 'LOC': loc_list_group, 
        'PER': per_list_group, 'DATE': date_list_group
    }

    augment_sentences_train = train_data.copy()
    for sentence in train_data:
        results_augment = augment_sentence(sentence, list_ent_aug=ents_groups)
        augment_sentences_train = results_augment + augment_sentences_train

    return augment_sentences_train
    

def list_to_pd_format(data):
    sent_id, ids, words, tags = 1, [], [], []
    for sent in data:
        for word in sent:
            if word[0] != '.':
                ids.append(sent_id)
                words.append(word[0])
                tags.append(word[1])
            else:
                ids.append(sent_id)
                words.append(word[0])
                tags.append(word[1])
                sent_id += 1
                
    dataframe = {"sentence_id": ids, "word": words, "iob_tag": tags}
    pd_iob_data = pd.DataFrame.from_dict(dataframe)
    return pd_iob_data