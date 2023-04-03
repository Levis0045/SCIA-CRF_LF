from iteration_utilities import unique_everseen
import pandas as pd

"""
In data augmentation, there are several methods depending on task, to augment lowest data.

- Word replacement (classification): using synonyms, antonyms, 
- Mention replacement (QnA): Raiman and Miller (2017) use external knownledge based
- Swap words (classification):  Wei and Zou (2019) randomly choose two words in the sentence and swap their positions
to augment text classification training sets
- Generative models: Yu et al. (2018) train a question answering model with data generated by backtranslation 
from a neural machine translation model


For NER augmentation last research:
------------------------------------

- Label-wise token replacement (LwTR) = word replacement + Mention replacement + synonyms re

- Synonym replacement outperforms other augmentation on average when transformer models are used, 
whereas mention replacement appears to be most effective for recurrent models.
- Second, applying all data augmentation methods together outperforms any single data augmentation
on average, although, when the complete training set is used, applying single data augmentation may
achieve better results
- Third, data augmentation techniques are more effective when the training sets are small.


"""

# function that read IOB file and build data structure for train, test and dev
def read_format_data(filename):
    sents_id, words, all_tags = [], [], []
    all_extracted_data = []
    with open(filename, encoding='utf-8') as iob:
        sentence, id_sent, tags = [], 1, []
        for line in iob:
            if len(line) > 1:
                word, tag = line.strip().split(' ')
                sentence.append((word, tag))
                sents_id.append(id_sent)
                words.append(word)
                all_tags.append(tag)
                tags.append(tag)
            else:
                if sentence[-1] != '.': 
                    sentence.append(('.', 'PUNCT'))
                    words.append('.')
                    all_tags.append('PUNCT')
                sents_id.append(id_sent)
                all_extracted_data.append(sentence)
                sentence, tags = [], []
                id_sent += 1
    dataframe = {"sentence_id": sents_id, "word": words, "tags": all_tags}
    pd_iob_data = pd.DataFrame.from_dict(dataframe)
    return all_extracted_data, pd_iob_data


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


""" 
#print(org_list[0:12])
#org_list_group   = extract_iob_group_entities(org_list, output_format='list', remove_duplicates=True)
#date_list_group = extract_iob_group_entities(date_list)
#loc_list_group  = extract_iob_group_entities(loc_list)
#per_list_group  = extract_iob_group_entities(per_list)
#org_list_group[:12]
#print(only_train_ner_data[10])


data_extract = [('Mdyə̂faʼ', 'O'), ('mtəŋláʼ', 'B-LOC'), ('shyə̂ŋkaʼ', 'I-LOC'), (',', 'O'), 
('təŋláʼ', 'B-LOC'), ('ŋkaʼ', 'I-LOC'), ('gə́', 'O'), ('təŋláʼ', 'O'), 
('Adamáwǎ', 'B-LOC'), ('kuʼ', 'O'), ('dəŋ', 'O'), ('é', 'O'), ('.', 'O'), ('.', 'O')]
#detect_iob_tag_position(data_extract)

#results_augment = augment_sentence(sent, list_ent_aug=ents_groups)
#for sent in results_augment: print('\t', " ".join([i[0] for i in sent]))
"""