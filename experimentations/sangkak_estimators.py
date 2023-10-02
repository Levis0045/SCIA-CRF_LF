
from pathlib import Path
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
import multiprocessing as mp


from collections import defaultdict
from pandas import DataFrame as pd_DataFrame, concat as pd_concat
from datetime import datetime

from features import read_pos_format_data, sent2labels, sent2features



__all__ = ['SangkakPosProjetReader', 'SangkakPosFeaturisation']


class SangkakPosProjetReader(TransformerMixin, BaseEstimator):

    def __init__(self, copy=True):
        self.copy = copy

    def fit(self, file_path, y=None):  
        self.file_path = file_path
        if Path(file_path).exists():
            self.file_path = file_path
            return self
        else: raise Exception("File not found: %s" % file_path)

    def transform(self, filename, copy=None, label=False):
        # check_is_fitted(self)
        _, pd_iob_data = read_pos_format_data(filename)
        
        return pd_iob_data

    def build_data_model(self, extract_data):
        group_by_entities = defaultdict(list)

        # group words of the sentence by tag
        words = []
        for sent in extract_data:
            for i, wd in enumerate(sent):
                if wd[0] not in words:
                    group_by_entities[wd[1]].append((wd, i))
                    words.append(wd[0])
                elif  wd[0] in words and wd[1] not in group_by_entities.keys():
                    group_by_entities[wd[1]].append((wd, i))
        del words

        # group tag by word and its position in sentence
        group_by_position_tag = {}
        for tag, list_values in dict(group_by_entities).items():
            group_by_position = defaultdict(list)
            for wd in list_values: group_by_position[wd[1]].append(wd[0][0])
            group_by_position_tag[tag] = group_by_position

        return group_by_position_tag
    
    def augment_sentences(self, input_sentences, limit=10, 
                         exclude_tags=["PUNCT"], exclude_words=[]):
        print(f"-> [{datetime.now()}] Augment input sentences with position to position algorithm")
        from random import choice, sample

        all_sentences = []
        augmented_sentences = {}

        # building data model on all word and position in the sentence
        group_by_position_tag = self.build_data_model(input_sentences)
        #print(group_by_position_tag.keys())
        # print('Number of sentences to augment: ', len(input_sentences))

        for sentence in input_sentences:
            # augment data sentence
            results_aug = []
            all_words = [x[0] for x in sentence]

            #print('\nmain sentence: \n', all_words, '\n')

            for p, word_tag in enumerate(sentence):
                # exclude defined input words
                if word_tag[1] in exclude_tags: continue
                if word_tag[0] in exclude_words: continue

                # only take concrete word in the sentence
                word_position   = group_by_position_tag[word_tag[1]]
                list_match_word = word_position.get(p)

                if not list_match_word: continue

                # print('\nlist_match_word: ', list_match_word)

                # delete words in position already in the main sentence
                clean_list_match_words = []
                for w in list_match_word: 
                    if w not in all_words: clean_list_match_words.append(w)
                del list_match_word

                # print('clean_list_match_words: ', clean_list_match_words)
                # if we cannot find word in the main sentence
                if len(clean_list_match_words) == 0: continue
                
                # stop with limit criteria
                if limit and limit <= len(clean_list_match_words): 
                    sample_list_match_words = sample(clean_list_match_words, k=limit)
                else: sample_list_match_words = clean_list_match_words.copy()                    

                #print(len(clean_list_match_words), len(sample_list_match_words))

                #for w in sample_list_match_words:
                w = choice(sample_list_match_words) 
                # copy the main sentence
                copy_sentence = sentence.copy()
                # take random word in the clean list of word matched
                # w = choice(clean_list_match_words)
                # delete current word position
                del copy_sentence[p]
                assert len(copy_sentence) < len(sentence)
                # insert another word from list in the same position
                #copy_sentence.insert(p-1, ("[", "PUNCT"))
                copy_sentence.insert(p, (w, word_tag[1]))
                #copy_sentence.insert(p+1, ("]", "PUNCT"))
                assert len(copy_sentence) == len(sentence)
                #if len(results_aug) <= 5: 
                sent_norm = " ".join([o[0] for o in copy_sentence])
                if sent_norm not in all_sentences:
                    #print(sentence, '\n', copy_sentence)
                    results_aug.append(copy_sentence)

                all_sentences.append(sent_norm)
                
                #print(f'\t--> generated sentences for word  : {word_tag[0]}')
                #print(f'\t--> number of sentences generated : {len(results_aug)}')
                # for x in results_aug: print("\t\t"," ".join([o[0] for o in x]))
                
            augmented_sentences[" ".join(all_words)] = [sentence]+results_aug

        return augmented_sentences

    def transform_analysis(self, augment=True, **kwargs):
        # check_is_fitted(self)
        print(f"-> [{datetime.now()}] Read input sentences")
        extracted_data, pd_iob_data = read_pos_format_data(self.file_path)
        if augment:
            augmented_data = self.augment_sentences(extracted_data,  **kwargs)
            ids, sents_id, words, tags = 1, [], [], []
            normalized_augmented_data = [x for y in augmented_data.values() for x in y]
            for sent in normalized_augmented_data:
                for w in sent:
                    words.append(w[0])
                    tags.append(w[1])
                    sents_id.append(ids)
                ids += 1
            dataframe = {"sentence_id": sents_id, "word": words, "tags": tags}
            return normalized_augmented_data, pd_DataFrame.from_dict(dataframe)
                    
        return extracted_data, pd_iob_data

    def _more_tags(self):
        return {"stateless": True}

def concat_processes(k, v, col):
    return pd_DataFrame([[k] + list(v.values())], columns=col)
        
class SangkakPosFeaturisation(TransformerMixin, 
                              BaseEstimator):

    def __init__(self, norm="l2", *, copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):  
        return self

    def transform(self, X, copy=None, label=False):
        # check_is_fitted(self)
        print(f"-> [{datetime.now()}] Featurisation of input {'train' if not label else 'label'} sentences")
        
        copy = copy if copy is not None else self.copy
        # X = self._validate_data(X, accept_sparse="csr", reset=False)
        if copy:
            from copy import deepcopy
            X_cp = deepcopy(X)
        else: X_cp = X

        data_sents_formated = [[word for word in sentence] for sentence in X_cp]
       
        if label:
            X_cp = [sent2labels(s) for s in data_sents_formated]
        else:
            X_cp = [sent2features(s) for s in data_sents_formated]
        return X_cp

    def transform_to_sagemaker_format(self, Xt, yt, label="train",
                                      normalize=None):
        print(f"-> [{label}][{datetime.now()}] Building sagemaker data for classification")            
        columns = ['labels'] + list(Xt[0][0].keys())
        
        core = 4
        pool = mp.Pool(core)
        
        trans_data = [(i, k) for x, y in zip(Xt, yt) for i, k in zip(x, y)]
        print("\t*using multiprocessing to loop: ", len(trans_data))
        results_set = [pool.apply_async(concat_processes, [k, v, columns]) for v, k in trans_data]
        del trans_data

        print(f"\t*get result async loop with '{core}' cores...")
        df = pd_concat([i.get(timeout=10) for i in results_set], ignore_index=True)
        print(df.describe())
        
        if normalize is not None and type(normalize) == dict:
            print("\tnormalize: ")
            categorical = normalize['categorical_features']
            numerical = normalize['numerical_features']

            df[categorical] = df[categorical].astype("category")
            assert len(list(df.select_dtypes(include="category").columns)) == len(categorical)
            df[numerical] = df[numerical].astype("int")
            assert len(list(df.select_dtypes(include="number").columns)) == len(numerical)

        return df

    def _more_tags(self):
        return {"stateless": True}