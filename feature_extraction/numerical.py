import math
import nltk
import os
import gensim.models.word2vec as w2v
import numpy as np


class TitleOverlap:
    def extract(self, tdf, tdf_un, ndf):
        ndf['title_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term'], tdf['product_title'])]

class DescriptionOverlap:
    def extract(self, tdf, tdf_un, ndf):
        ndf['descr_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term'], tdf['product_description'])]

class DescriptionOverlapJaccard:
    def extract(self, tdf, tdf_un, ndf):
        tmp = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term'], tdf['product_description'])]
        ndf['descr_overlap_jc'] = [z / (len(x.split()) + len(y.split()) - z)  for x,y,z in zip(tdf['search_term'], tdf['product_description'], tmp)]

class DescriptionMatch:
    def extract(self, tdf, tdf_un, ndf):
        ndf['description_match'] = [1 if x in y else 0 for x,y in zip(tdf['search_term'], tdf['product_description'])]

        
        
class TitleOverlap2gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['title_overlap_2gram'] = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_2gram'], tdf['product_title_2gram'])]

class TitleOverlap2gramJaccard:
    def extract(self, tdf, tdf_un, ndf):
        tmp = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_2gram'], tdf['product_title_2gram'])]
        ndf['title_overlap_2gram_jaccard'] = [z / (len(x.split()) + len(y.split()) - z)  for x,y,z in zip(tdf['search_term_2gram'], tdf['product_title_2gram'], tmp)]

class TitleMatch2gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['title_match_2gram'] = [1 if x in y else 0 for x,y in zip(tdf['search_term_2gram'], tdf['product_title_2gram'])]
        
class TitleOverlap3gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['title_overlap_3gram'] = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_3gram'], tdf['product_title_3gram'])]

class TitleOverlap3gramJaccard:
    def extract(self, tdf, tdf_un, ndf):
        tmp = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_3gram'], tdf['product_title_3gram'])]
        ndf['title_overlap_3gram_jaccard'] = [z / (len(x.split()) + len(y.split()) - z)  for x,y,z in zip(tdf['search_term_3gram'], tdf['product_title_3gram'], tmp)]

class TitleMatch3gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['title_match_3gram'] = [1 if x in y else 0 for x,y in zip(tdf['search_term_3gram'], tdf['product_title_3gram'])]
        
class TitleOverlap4gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['title_overlap_4gram'] = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_4gram'], tdf['product_title_4gram'])]

class TitleOverlap4gramJaccard:
    def extract(self, tdf, tdf_un, ndf):
        tmp = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_4gram'], tdf['product_title_4gram'])]
        ndf['title_overlap_4gram_jaccard'] = [z / (len(x.split()) + len(y.split()) - z)  for x,y,z in zip(tdf['search_term_4gram'], tdf['product_title_4gram'], tmp)]

class TitleMatch4gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['title_match_4gram'] = [1 if x in y else 0 for x,y in zip(tdf['search_term_4gram'], tdf['product_title_4gram'])]
        


class BrandMatch:
    def extract(self, tdf, tdf_un, ndf):
        ndf['brand_match'] = [1 if str(y) in x else 0 for x,y in zip(tdf['search_term'], tdf['brand'])]

class ColorOverlap:
    def extract(self, tdf, tdf_un, ndf):
        ndf['color_overlap'] = [sum(int(word in str(y)) for word in x.split()) for x,y in zip(tdf['search_term'], tdf['colors'])]

class ColorMatch:
    def extract(self, tdf, tdf_un, ndf):
        ndf['color_match'] = [1 if str(y) in x else 0 for x,y in zip(tdf['search_term'], tdf['colors'])]

class QueryLength:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_length'] = [len(x.split()) for x in tdf['search_term']]

class QueryLength2gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_length_2gram'] = [len(x.split()) for x in tdf['search_term_2gram']]

class QueryLength3gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_length_3gram'] = [len(x.split()) for x in tdf['search_term_3gram']]
        
class QueryLength4gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_length_4gram'] = [len(x.split()) for x in tdf['search_term_4gram']]
        
class QueryCharachterLength:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_character_length'] = [len(x) for x in tdf['search_term']]

class QueryAverageWordLength:
    def extract(self, tdf, tdf_un, ndf):
        query_length = [len(x.split()) for x in tdf['search_term']]
        query_char_length = ndf['query_character_length'] = [len(x) for x in tdf['search_term']]
        ndf['query_average_word_length'] = [y/x for x,y in zip(query_length, query_char_length)]

class Ratio2gramsInQueryMatchInTitle:
    def extract(self, tdf, tdf_un, ndf):
        query_ngram_length = [len(x.split()) for x in tdf['search_term_2gram']]
        title_ngram_overlap = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_2gram'], tdf['product_title_2gram'])]
        ndf['total_match_title'] = [math.floor(x/y) for x,y in zip(title_ngram_overlap, query_ngram_length)]

class AmountOfNumbersInQuery:
    def extract(self, tdf, tdf_un, ndf):
        ndf['amount_numbers'] = [sum(s.isdigit() for s in x.split()) for x in tdf['search_term']]

class RatioNumbersInQuery:
    def extract(self, tdf, tdf_un, ndf):
        query_char_length = [len(x) for x in tdf['search_term']]
        amount_numbers = [sum(s.isdigit() for s in x.split()) for x in tdf['search_term']]
        ndf['ratio_numbers'] = [y/x for x,y in zip(query_char_length, amount_numbers)]

class NumberOfNouns:
    def extract(self, tdf, tdf_un, ndf):
        sentencelist = [x.split() for x in tdf['search_term']]
        result = nltk.pos_tag_sents(sentencelist)
        nouns = [[word for word,pos in lst if pos in ['NN', 'NNP', 'NNS', 'NNPS']] for lst in result]
        ndf['number_of_nouns'] = [int(len(x)) for x in nouns]

class SpellingCorrectionPerformed:
    def extract(self, tdf, tdf_un, ndf):
        ndf['spell_corrected'] = [x for x in tdf['spell_corrected']]

class Word2VecSimilarity:
    def extract(self, tdf, tdf_un, ndf):
        if os.path.isfile('data/word2vec/full'):
            model = w2v.Word2Vec.load('data/word2vec/full')
            title_in_model = [[q for q in x.split() if q in model.vocab] for x in tdf['product_title']]
            query_in_model = [[q for q in x.split() if q in model.vocab] for x in tdf['search_term']]

            ndf['word2vec_sim'] = [
                model.n_similarity(x, y) if (len(x) > 0 and len(y) > 0) else 0
                for x,y in zip(query_in_model, title_in_model)
            ]

            
class Word2VecSimilarityPretrained:
    def extract(self, tdf, tdf_un, ndf):
        if os.path.isfile('data/word2vec/GoogleNews.bin'):
            model = w2v.Word2Vec.load_word2vec_format('data/word2vec/GoogleNews.bin', binary=True)
            ndf['word2vec_pre'] = [
                sum(
                    model.similarity(q, t)
                    for q in x.split() if q in model.vocab
                    for t in y.split() if t in model.vocab
                )
                for x,y in zip(tdf_un['search_term'], tdf_un['product_title'])
            ]

class NumberOfVowelsSearchTerm:
    def extract(self, tdf, tdf_un, ndf):
        ndf['num_vovels_search_term'] = [len([y for y in x if y in 'aeouiy']) for x in tdf['search_term']]

class NumberOfVowelsTitle:
    def extract(self, tdf, tdf_un, ndf):
        ndf['num_vovels_title'] = [len([y for y in x if y in 'aeouiy']) for x in tdf['product_title']]

        
class AveragePositionMatchedSearchTerms:
    def extract(self, tdf, tdf_un, ndf):
        positions = [np.mean([a for a,b in enumerate(y.split()) if (b in x.split())]) for x,y in zip(tdf['search_term'], tdf['product_title'])]
        
        ndf['avg_pos_terms'] = [0 if math.isnan(x) else x for x in positions]
        

class DistanceMatchedSearchTerms:
    def extract(self, tdf, tdf_un, ndf):
        positions = [([a for a,b in enumerate(y.split()) if (b in x.split())]) for x,y in zip(tdf['search_term'], tdf['product_title'])]
        
        ndf['dist_matched_terms'] = [0 if len(x)<2 else np.abs(x[0]-x[1]) for x in positions]   
        
class LastWordInTitle:
    def extract(self, tdf, tdf_un, ndf):       
        ndf['last_word_match'] = [1 if (y.split()[-1] in x) else 0  for x,y in zip(tdf['search_term'], tdf['product_title'])]
        
class FirstWordInTitle:
    def extract(self, tdf, tdf_un, ndf):       
        ndf['first_word_match'] = [1 if (y.split()[0] in x) else 0  for x,y in zip(tdf['search_term'], tdf['product_title'])]
        
