import math
import nltk

class DescriptionOverlap:
    def extract(self, tdf, ndf):
        ndf['descr_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term'], tdf['product_description'])]

class DescriptionOverlapJaccard:
    def extract(self, tdf, ndf):
        tmp = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term'], tdf['product_description'])]
        ndf['descr_overlap_jc'] = [z / (len(x.split()) + len(y.split()) - z)  for x,y,z in zip(tdf['search_term'], tdf['product_description'], tmp)]

class DescriptionMatch:
    def extract(self, tdf, ndf):
        ndf['description_match'] = [1 if x in y else 0 for x,y in zip(tdf['search_term'], tdf['product_description'])]

class TitleOverlapNgram:
    def extract(self, tdf, ndf):
        ndf['title_overlap_ngram'] = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_ngram'], tdf['product_title_ngram'])]

class TitleOverlapNgramJaccard:
    def extract(self, tdf, ndf):
        tmp = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_ngram'], tdf['product_title_ngram'])]
        ndf['title_overlap_ngram_jaccard'] = [z / (len(x.split()) + len(y.split()) - z)  for x,y,z in zip(tdf['search_term_ngram'], tdf['product_title_ngram'], tmp)]

class TitleMatchNgram:
    def extract(self, tdf, ndf):
        ndf['title_match_ngram'] = [1 if x in y else 0 for x,y in zip(tdf['search_term_ngram'], tdf['product_title_ngram'])]

class BrandMatch:
    def extract(self, tdf, ndf):
        ndf['brand_match'] = [1 if str(y) in x else 0 for x,y in zip(tdf['search_term'], tdf['brand'])]

class ColorOverlap:
    def extract(self, tdf, ndf):
        ndf['color_overlap'] = [sum(int(word in str(y)) for word in x.split()) for x,y in zip(tdf['search_term'], tdf['colors'])]

class ColorMatch:
    def extract(self, tdf, ndf):
        ndf['color_match'] = [1 if str(y) in x else 0 for x,y in zip(tdf['search_term'], tdf['colors'])]

class QueryLength:
    def extract(self, tdf, ndf):
        ndf['query_length'] = [len(x.split()) for x in tdf['search_term']]

class QueryLengthNgram:
    def extract(self, tdf, ndf):
        ndf['query_length_ngram'] = [len(x.split()) for x in tdf['search_term_ngram']]

class QueryCharachterLength:
    def extract(self, tdf, ndf):
        ndf['query_character_length'] = [len(x) for x in tdf['search_term']]

class QueryAverageWordLength:
    def extract(self, tdf, ndf):
        query_length = [len(x.split()) for x in tdf['search_term']]
        query_char_length = ndf['query_character_length'] = [len(x) for x in tdf['search_term']]
        ndf['query_average_word_length'] = [y/x for x,y in zip(query_length, query_char_length)]

class RatioNgramsInQueryMatchInTitle:
    def extract(self, tdf, ndf):
        query_ngram_length = [len(x.split()) for x in tdf['search_term_ngram']]
        title_ngram_overlap = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_ngram'], tdf['product_title_ngram'])]
        ndf['total_match_title'] = [math.floor(x/y) for x,y in zip(title_ngram_overlap, query_ngram_length)]

class AmountOfNumbersInQuery:
    def extract(self, tdf, ndf):
        ndf['amount_numbers'] = [sum(s.isdigit() for s in x.split()) for x in tdf['search_term']]

class RatioNumbersInQuery:
    def extract(self, tdf, ndf):
        query_char_length = [len(x) for x in tdf['search_term']]
        amount_numbers = [sum(s.isdigit() for s in x.split()) for x in tdf['search_term']]
        ndf['ratio_numbers'] = [y/x for x,y in zip(query_char_length, amount_numbers)]

class NumberOfNouns:
    def extract(self, tdf, ndf):
        sentencelist = [x.split() for x in tdf['search_term']]
        result = nltk.pos_tag_sents(sentencelist)
        nouns = [[word for word,pos in lst if pos in ['NN', 'NNP', 'NNS', 'NNPS']] for lst in result]

        ndf['number_of_nouns'] = [int(len(x)) for x in nouns]

class SpellingCorrectionPerformed:
    def extract(self, tdf, ndf):
        ndf['spell_corrected'] = [x for x in tdf['spell_corrected']]