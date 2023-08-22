
import string
import re
import unicodedata
from pandas import DataFrame as pd_DataFrame
# l'ajout des tags suivants au mot courant améliore significativement le modèle
# l'ajout des informations sur les tons

bantou_tones = [f"{x} " for x in " ́̄̀̌̂" if x != " "]
string_tones = "".join(bantou_tones)
tones_search = re.compile(string_tones)
bantou_letters = string.ascii_letters+"ǝɔᵾɓɨşœɑʉɛɗŋøẅëïə"

def word_decomposition(input_str):
    """Decompse input string in to words

    Args:
        input_str (str): input string

    Returns:
        str: input string
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    word_decomp = " ".join([x for x in nfkd_form ])
    return word_decomp

def extract_tone(input_str):
    """Extract tone from input string

    Args:
        input_str (str): input string

    Returns:
        str: tones found from input string
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    #print([x for x in nfkd_form if x not in string.ascii_letters])
    tones = [x for x in nfkd_form if x not in bantou_letters]
    str_tones =  "".join(list(set([x.strip() for x in tones 
                     if x in " ̄ ̀ ̌ ̂ '"])))
    # print(str_tones)
    return str_tones if len(str_tones) != 0 else None
    
def number_tone_word(input_str):
    """Get number of tone found in the input string

    Args:
        input_str (str): input string

    Returns:
        int: number of tone found in the input string
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    tone_str = [x for x in nfkd_form if x not in bantou_letters]

    return len([x.strip() for x in tone_str 
                     if x not in ['.', 'Ŋ', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    
def word2features(sent, i):
    word = sent[i][0]
    len_tone = number_tone_word(word)
    tones = extract_tone(word)
    features = {
        'word': word,
        #'bias': 1.0,
        'word.tones': tones if tones else "",
        'word.normalized': unicodedata.normalize('NFKD', word),
        'word.position': i,
        'word.has_hyphen': int('-' in word),
        'word.lower()': word.lower(),
        'word.start_with_capital': int(word[0].isupper()) if i > 0 else -1,
        'word.have_tone': 1 if len_tone>0 else 0,
        'word.prefix': word[:2] if len(word)>2 else "",
        'word.root': word[3:] if len(word)>2 else "",
        'word.ispunctuation': int(word in string.punctuation),
        'word.isdigit()': int(word.isdigit()),
        'word.EOS': 1 if word in ['.','?','!'] else 0,
        'word.BOS': 1 if i == 0 else 0,
        '-1:word': sent[i-1][0] if i > 0 else "",
        '-1:word.position': i-1 if i > 0 else -1,
        '-1:word.tag': sent[i-1][1] if i > 0 else "",
        #'-1:word.letters': word_decomposition(sent[i-1][0]) if i > 0 else -1,
        '-1:word.normalized': unicodedata.normalize('NFKD', sent[i-1][0]) if i > 0 else "",
        '-1:word.start_with_capital': int(sent[i-1][0][0].isupper()) if i > 0 else -1,
        '-1:len(word-1)': len(sent[i-1][0]) if i > 0 else -1,
        '-1:word.lower()': sent[i-1][0].lower() if i > 0 else "",
        '-1:word.isdigit()': int(sent[i-1][0].isdigit()) if i > 0 else -1,
        '-1:word.ispunctuation': int((sent[i-1][0] in string.punctuation)) if i > 0 else 0,
        '-1:word.BOS': 1 if (i-1) == 0 else 0,
        '-1:word.EOS': 1 if i > 0 and sent[i-1][0] in ['.','?','!'] else 0,
        '+1:word': sent[i+1][0] if i < len(sent)-1 else "",
        '+1:word.tag': sent[i+1][1] if i < len(sent)-1 else "",
        '+1:word.position': i+1,
        #'+1:word.letters': word_decomposition(sent[i+1][0]) if i < len(sent)-1 else -1,
        '+1:word.normalized': unicodedata.normalize('NFKD', sent[i+1][0]) if i < len(sent)-1 else "",
        '+1:word.start_with_capital': int(sent[i+1][0][0].isupper()) if i < len(sent)-1 else -1,
        '+1:len(word+1)': len(sent[i+1][0]) if i < len(sent)-1 else -1,
        '+1:word.lower()': sent[i+1][0].lower() if i < len(sent)-1 else "",
        '+1:word.isdigit()': int(sent[i+1][0].isdigit()) if i < len(sent)-1 else -1,
        '+1:word.ispunctuation': int((sent[i+1][0] in string.punctuation)) if i < len(sent)-1 else -1,
        '+1:word.BOS': 1 if i < 0 else 0,
        '+1:word.EOS': 1 if i < len(sent)-1 and sent[i+1][0] in ['.','?','!'] else 0
    }

    # if tagword not in ['B-ORG','B-LOC']: features.update({'-1:word.tag()': tagword1})
    
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [word[1] for word in sent]

def sent2tokens(sent):
    return [word[0] for word in sent]

def read_pos_format_data(filename):
    sents_id, words, all_tags = [], [], []
    all_extracted_data = []
    with open(filename, encoding='utf-8') as iob:
        sentence, id_sent, tags = [], 1, []
        for line in iob:
            if len(line) > 1:
                word, tag = line.strip().split(' ')
                sentence.append((word.strip(), tag))
                sents_id.append(id_sent)
                words.append(word.strip())
                all_tags.append(tag)
                tags.append(tag)
            else:
                if sentence[-1][0] not in ['.','!','?']: 
                    # normalized punctuation at the end of all sentences
                    sents_id.append(id_sent)
                    sentence.append(('.', 'PUNCT'))
                    words.append('.')
                    all_tags.append('PUNCT')
                
                all_extracted_data.append(sentence)
                sentence, tags = [], []
                id_sent += 1
    print(len(sents_id),len(words), len(all_tags))
    dataframe = {"sentence_id": sents_id, "word": words, "tags": all_tags}
    pd_iob_data = pd_DataFrame.from_dict(dataframe)
    return all_extracted_data, pd_iob_data

def format_data(csv_data):
    sents = []
    """for i in range(len(csv_data)):
        if math.isnan(float(csv_data.iloc[i, 0])): continue
        elif csv_data.iloc[i, 0] == 1.0:
            sents.append([[csv_data.iloc[i, 1], csv_data.iloc[i, 2]]])
        else:
            try: sents[-1].append([csv_data.iloc[i, 1], csv_data.iloc[i, 2]])
            except: print('...', csv_data.iloc[i, 2])
    for sent in sents:
        for i, word in enumerate(sent):
            if type(word[0]) != str:
                del sent[i]"""
    return csv_data