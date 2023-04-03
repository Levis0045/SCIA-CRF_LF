
import string
import re
import unicodedata

# l'ajout des tags suivants au mot courant améliore significativement le modèle
# l'ajout des informations sur les tons

bantou_tones = [f"{x} " for x in " ́̄̀̌̂" if x != " "]
string_tones = "".join(bantou_tones)
tones_search = re.compile(string_tones)

bantou_letters = string.ascii_letters+"ǝɔᵾɓɨşœɑʉɛɗŋøẅëïə"

def remove_accents(input_str):
    """Remove accents from input string in other to get ascii string

    Args:
        input_str (str): input string

    Returns:
        str: output ascii string
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    #print([x for x in nfkd_form if x not in string.ascii_letters])
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode('utf8')

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
    return " ".join(tones)

def number_tone_word(input_str):
    """Get number of tone found in the input string

    Args:
        input_str (str): input string

    Returns:
        int: number of tone found in the input string
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    len_tone_str = len([x for x in nfkd_form if x not in bantou_letters])
    return len_tone_str

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

def compare_two_words(input_str1, input_str2):
    """Compare two words

    -- in progress

    Args:
        input_str1 (str): _description_
        input_str2 (str): _description_

    Returns:
        _type_: _description_
    """
    nfkd_form1 = unicodedata.normalize('NFKD', input_str1)
    nfkd_form2 = unicodedata.normalize('NFKD', input_str2)
    len_tone_str1 = len([x for x in nfkd_form1 if x not in bantou_letters])
    len_tone_str2 = len([x for x in nfkd_form2 if x not in bantou_letters])
    only_ascii = nfkd_form1.encode('ASCII', 'ignore')
    return only_ascii.decode('utf8')

def word2features(sent, i):
    word = sent[i][0]
    tagword = sent[i][1]
    len_tone = number_tone_word(word)
    len_word = len(word) / 2
    features = {
        'bias': 1.0,
        'word': word,
        'word.tones': extract_tone(word),
        'word.normalized': unicodedata.normalize('NFKD', word),
        #'word.letters': word_decomposition(word),
        'word.position': i,
        #'word[:3]': word[:3], impacte négativement les résultats
        #'word[:2]': word[:2],
        #'word[-3:]': word[-3:],
        #'word[-2:]': word[-2:],
        #'word.middle_start': word[:int(len_word)],
        #'word.middle_end': word[int(len_word):],
        'word.has_hyphen': '-' in word,
        #'word.unaccent': remove_accents(word),
        'word.lower()': word.lower(),
        'word.start_with_capital': word[0].isupper(),
        'word.have_tone': True if len_tone>0 else False,
        #'word.len_tones': len_tone,
        'word.ispunctuation': (word in string.punctuation),
        'word.isdigit()': word.isdigit()
    }
    # if word == '.': features['EOS'] = True
    
    """
    if i > 0:
        word1 = sent[i-1][0]
        tagword1 = sent[i-1][1]
        len_tone1 = number_tone_word(word1)
        len_word1 = len(word1) / 2
        features.update({
            '-1:word': word1,
            '-1:word.position': i-1,
            '-1:word.letters': word_decomposition(word1),
            '-1:word.normalized': unicodedata.normalize('NFKD', word1),
            '-1:word.start_with_capital': word1[0].isupper(),
            '-1:len(word1)': len(word1),
            '-1:word.lower()': word1.lower(),
            #'-1:word.tag()': tagword1,
            #'-1:word.unaccent': remove_accents(word1),
            #'-1:word.middle_start': word1[:int(len_word1)],
            #'-1:word.middle_end': word1[int(len_word1):],
            #'-1:word.have_tone': True if len_tone1>0 else False,
            #'-1:word.len_tones': len_tone1,
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.ispunctuation': (word1 in string.punctuation)
        })
        if tagword not in ['B-ORG','B-LOC']: features.update({'-1:word.tag()': tagword1})
    else: features['BOS'] = True
    
    if i > 1:
        word2 = sent[i-2][0]
        tagword2 = sent[i-2][1]
        len_tone2 = number_tone_word(word2)
        len_word2 = len(word2) / 2
        features.update({
            '-2:word': word2,
            '-2:word.position': i-2,
            '-2:word.letters': word_decomposition(word2),
            '-2:word.normalized': unicodedata.normalize('NFKD', word2),
            '-2:word.start_with_capital': word2[0].isupper(),
            '-2:len(word2)': len(word2),
            '-2:word.lower()': word2.lower(),
            '-2:word.tag()': tagword2,
            #'-2:word.unaccent': remove_accents(word2),
            #'-2:word.middle_start': word2[:int(len_word2)],
            #'-2:word.middle_end': word2[int(len_word2):],
            #'-2:word.have_tone': True if len_tone2>0 else False,
            #'-2:word.len_tones': len_tone2,
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.ispunctuation': (word2 in string.punctuation)
        })

    if i > 2:
        word3 = sent[i-3][0]
        tagword3 = sent[i-3][1]
        len_tone3 = number_tone_word(word3)
        len_word3 = len(word3) / 2
        features.update({
            '-3:word': word3,
            '-3:word.position': i+3,
            '-3:word.letters': word_decomposition(word3),
            '-3:word.normalized': unicodedata.normalize('NFKD', word3),
            '-3:word.start_with_capital': word3[0].isupper(),
            '-3:len(word3)': len(word3),
            '-3:word.tag()': tagword3,
            #'-3:word.lower()': word2.lower(),
            #'-3:word.unaccent': remove_accents(word2),
            #'-3:word.middle_start': word2[:int(len_word2)],
            #'-3:word.middle_end': word2[int(len_word2):],
            #'-3:word.have_tone': True if len_tone2>0 else False,
            #'-3:word.len_tones': len_tone2,
            '-3:word.isdigit()': word3.isdigit(),
            '-3:word.ispunctuation': (word3 in string.punctuation)
        })
    
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        tagword1 = sent[i+1][1]
        len_tone1 = number_tone_word(word1)
        len_word1 = len(word1) / 2
        features.update({
            '+1:word': word1,
            '+1:word.position': i+1,
            '+1:word.letters': word_decomposition(word1),
            '+1:word.normalized': unicodedata.normalize('NFKD', word1),
            '+1:word.start_with_capital': word1[0].isupper(),
            '+1:len(word2)': len(word1),
            '+1:word.lower()': word1.lower(),
            #'+1:word.tag()': tagword1,
            #'+1:word.unaccent': remove_accents(word1),
            #'+1:word.middle_start': word1[:int(len_word1)],
            #'+1:word.middle_end': word1[int(len_word1):],
            #'+1:word.have_tone': True if len_tone1>0 else False,
            #'+1:word.len_tones': len_tone1,
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.ispunctuation': (word1 in string.punctuation)
        })
    
    if i < len(sent)-2:
        word2 = sent[i+2][0]
        tagword2 = sent[i+2][1]
        len_tone2 = number_tone_word(word2)
        len_word2 = len(word2) / 2
        features.update({
            '+2:word': word2,
            '+2:word.position': i+2,
            '+2:word.letters': word_decomposition(word2),
            '+2:word.normalized': unicodedata.normalize('NFKD', word2),
            '+2:word.start_with_capital': word2[0].isupper(),
            '+2:len(word2)': len(word2),
            '+2:word.tag()': tagword2,
            #'+2:word.lower()': word2.lower(),
            #'+2:word.unaccent': remove_accents(word2),
            #'+2:word.middle_start': word2[:int(len_word2)],
            #'+2:word.middle_end': word2[int(len_word2):],
            #'+2:word.have_tone': True if len_tone2>0 else False,
            #'+2:word.len_tones': len_tone2,
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word.ispunctuation': (word2 in string.punctuation)
        })
        
    if i < len(sent)-3:
        word3 = sent[i+3][0]
        tagword3 = sent[i+3][1]
        len_tone3 = number_tone_word(word3)
        len_word3 = len(word3) / 2
        features.update({
            '+3:word': word3,
            '+3:word.position': i+3,
            '+3:word.letters': word_decomposition(word3),
            '+3:word.normalized': unicodedata.normalize('NFKD', word3),
            '+3:word.start_with_capital': word3[0].isupper(),
            '+3:len(word3)': len(word3),
            '+3:word.tag()': tagword3,
            #'+2:word.lower()': word2.lower(),
            #'+2:word.unaccent': remove_accents(word2),
            #'+2:word.middle_start': word2[:int(len_word2)],
            #'+2:word.middle_end': word2[int(len_word2):],
            #'+2:word.have_tone': True if len_tone2>0 else False,
            #'+2:word.len_tones': len_tone2,
            '+3:word.isdigit()': word3.isdigit(),
            '+3:word.ispunctuation': (word3 in string.punctuation)
        })
    """
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [word[1] for word in sent]

def sent2tokens(sent):
    return [word[0] for word in sent]

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