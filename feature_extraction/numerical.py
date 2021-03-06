## this is feature_extraction/numerical.py

import math
import nltk
import os
import gensim.models.word2vec as w2v
import re
import pickle
import numpy as np


########
### Q-P PAIR FEATURES
######## 

# title 
class TitleOverlap:
    def extract(self, tdf, tdf_un, ndf):
        ndf['title_overlap'] = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term'], tdf['product_title'])]

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
        
class Ratio2gramsInQueryMatchInTitle:
    def extract(self, tdf, tdf_un, ndf):
        query_2gram_length = [len(x.split()) for x in tdf['search_term_2gram']]
        title_2gram_overlap = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_2gram'], tdf['product_title_2gram'])]
        ndf['total_match_title'] = [math.floor(x/y) for x,y in zip(title_2gram_overlap, query_2gram_length)]
        
class Ratio3gramsInQueryMatchInTitle:
    def extract(self, tdf, tdf_un, ndf):
        query_3gram_length = [len(x.split()) for x in tdf['search_term_3gram']]
        title_3gram_overlap = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_3gram'], tdf['product_title_3gram'])]
        ndf['total_match_title'] = [math.floor(x/y) for x,y in zip(title_3gram_overlap, query_3gram_length)]

class Ratio4gramsInQueryMatchInTitle:
    def extract(self, tdf, tdf_un, ndf):
        query_4gram_length = [len(x.split()) for x in tdf['search_term_4gram']]
        title_4gram_overlap = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_4gram'], tdf['product_title_4gram'])]
        ndf['total_match_title'] = [math.floor(x/y) for x,y in zip(title_4gram_overlap, query_4gram_length)]

# % of query that match with title in terms of words, in terms of biwords
# ....


## word2vec similarities

# word2vec similarity of query and product title
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

class Word2vecSummedSimilarity:
    def extract(self, tdf, tdf_un, ndf):
        if os.path.isfile('data/word2vec/full'):
            model = w2v.Word2Vec.load('data/word2vec/full')
            title_in_model = [[q for q in x.split() if q in model.vocab] for x in tdf['product_title']]
            query_in_model = [[q for q in x.split() if q in model.vocab] for x in tdf['search_term']]

            ndf['word2vec_sumsim'] = [
                sum(
                    model.similarity(q, t)
                    for q in x.split() if q in model.vocab
                    for t in y.split() if t in model.vocab
                )
                for x,y in zip(tdf['search_term'], tdf['product_title'])
            ]
            
class Word2VecSummedSimilarityPretrained:
    def extract(self, tdf, tdf_un, ndf):
        if os.path.isfile('data/word2vec/GoogleNews.bin'):
            model = w2v.Word2Vec.load_word2vec_format('data/word2vec/GoogleNews.bin', binary=True)
            ndf['word2vec_sumpre'] = [
                sum(
                    model.similarity(q, t)
                    for q in x.split() if q in model.vocab
                    for t in y.split() if t in model.vocab
                )
                for x,y in zip(tdf_un['search_term'], tdf_un['product_title'])
            ]

class Word2VecSimilarityPretrained:
    def extract(self, tdf, tdf_un, ndf):
        if os.path.isfile('data/word2vec/GoogleNews.bin'):
            model = w2v.Word2Vec.load_word2vec_format('data/word2vec/GoogleNews.bin', binary=True)
            title_in_model = [[q for q in x.split() if q in model.vocab] for x in tdf_un['product_title']]
            query_in_model = [[q for q in x.split() if q in model.vocab] for x in tdf_un['search_term']]

            ndf['word2vec_pre'] = [
                model.n_similarity(x, y) if (len(x) > 0 and len(y) > 0) else 0
                for x, y in zip(query_in_model, title_in_model)
            ]


## description

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



## misc

class BrandMatch:
    def extract(self, tdf, tdf_un, ndf):
        ndf['brand_match'] = [1 if str(y) in x else 0 for x,y in zip(tdf['search_term'], tdf['brand'])]

class ColorOverlap:
    def extract(self, tdf, tdf_un, ndf):
        ndf['color_overlap'] = [sum(int(word in str(y)) for word in x.split()) for x,y in zip(tdf['search_term'], tdf['colors'])]

class ColorMatch:
    def extract(self, tdf, tdf_un, ndf):
        ndf['color_match'] = [1 if str(y) in x else 0 for x,y in zip(tdf['search_term'], tdf['colors'])]        












#########
#### QUERY FEATURES
#########

# these helper functions are used in the classes below
def charcount(x):
    digits = sum(c.isdigit() for c in x)
    chars   = sum(c.isalpha() for c in x)
    spaces  = sum(c.isspace() for c in x)
    others  = len(x) - digits - chars - spaces
    return digits,chars,spaces,others

def countdigits(x):
    digits, chars, spaces, others = charcount(x)
    return digits
def countchars(x):
    digits, chars, spaces, others = charcount(x)
    return chars
def countspaces(x):
    digits, chars, spaces, others = charcount(x)
    return spaces
def countothers(x):
    digits, chars, spaces, others = charcount(x)
    return others
def alpha_num_ratio(x):
    numericals, alphas, spaces, others = charcount(x)
    if numericals == 0:
        return 0
    else:
        alpha_num_ratio = (alphas)/(numericals)
        return alpha_num_ratio


## various length features

# query feature
class QueryLengthByTokens:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_token_length'] = [len(x.split()) for x in tdf['search_term']]

# query feature
class QueryLength2gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_length_2gram'] = [len(x.split()) for x in tdf['search_term_2gram']]

# query feature
class QueryLength3gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_length_3gram'] = [len(x.split()) for x in tdf['search_term_3gram']]

# query feature        
class QueryLength4gram:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_length_4gram'] = [len(x.split()) for x in tdf['search_term_4gram']]

# query feature
class QueryLengthByCharacters:
    def extract(self, tdf, tdf_un, ndf):
        ndf['query_character_length'] = [len(x) for x in tdf['search_term']]

# query feature
class QueryAverageTokenLength:
    def extract(self, tdf, tdf_un, ndf):
        query_length = [len(x.split()) for x in tdf['search_term']]
        query_char_length = ndf['query_character_length'] = [len(x) for x in tdf['search_term']]
        ndf['query_average_token_length'] = [y/x for x,y in zip(query_length, query_char_length)]

# query feature
# average nonword length
class QueryAverageNonwordLength:
    def extract(self, tdf, tdf_un, ndf):
        query_length = [len(x.split()) for x in tdf['search_term']]
        query_char_length = ndf['query_character_length'] = [len(x) for x in tdf['search_term']]
        ndf['query_average_token_length'] = [y/x for x,y in zip(query_length, query_char_length)]

# query feature
# absolute number of special characters in query
class LengthNonSpaceNonAlphaNonNumericalChars:

    def count_nonspacenonalphanonnumerical(self, x):
        numericals, alphas, spaces, others = charcount(x)
        return others

    def extract(self, tdf, tdf_un, ndf):
        ndf['query_length_nonspacenonalphanonnum_chars'] = [self.count_nonspacenonalphanonnumerical(x) for x in tdf['search_term']]

# query feature
# absolute number of mixed (not purely-alphabetical or purely-numerical) tokens in query
class LengthNonSpaceNonAlphaNonNumericalTokens:

    def count_nonspacenonalphanonnumerical(self, x):
        numericals, alphas, spaces, others = charcount(x)
        return others

    def extract(self, tdf, tdf_un, ndf):
        ndf['query_length_nonspacenonalphanonnum_tokens'] = [self.count_nonspacenonalphanonnumerical(nltk.word_tokenize((x))) for x in tdf['search_term']]

# query feature
# number of nouns in query by nltk POS-tags
class NumberOfNouns:
    def extract(self, tdf, tdf_un, ndf):
        sentencelist = [x.split() for x in tdf['search_term']]
        result = nltk.pos_tag_sents(sentencelist)
        nouns = [[word for word,pos in lst if pos in ['NN', 'NNP', 'NNS', 'NNPS']] for lst in result]
        ndf['number_of_nouns'] = [int(len(x)) for x in nouns]

# query feature
# number of adjectives in query? etc etc (other POS-tags)


# query feature
class NumberOfVowelsSearchTerm:
    def extract(self, tdf, tdf_un, ndf):
        ndf['num_vovels_search_term'] = [len([y for y in x if y in 'aeouiy']) for x in tdf['search_term']]

# query feature
## absolute occurrence of character class (numerical, alphabetical, spaces, others (signs etc.))
class CountsOfCharsPerClass:

    def extract(self, tdf, tdf_un, ndf):
        ndf['count_of_num_chars'] = [countdigits(x) for x in tdf['search_term']]
        ndf['count_of_alph_chars'] = [countchars(x) for x in tdf['search_term']]
        ndf['count_of_space_chars'] = [countspaces(x) for x in tdf['search_term']]
        ndf['count_of_other_chars'] = [countothers(x) for x in tdf['search_term']]

# query feature
## absolute occurrence of token class (numerical only, alphabethical only, mixed only)
class CountsOfTokensPerClass:
    def extract(self, tdf, tdf_un, ndf):
        ndf['count_of_pure_num_tokens'] = [countdigits(nltk.word_tokenize((x))) for x in tdf['search_term']]
        ndf['count_of_pure_alph_tokens'] = [countchars(nltk.word_tokenize((x))) for x in tdf['search_term']]
        ndf['count_of_other_tokens'] = [countothers(nltk.word_tokenize((x))) for x in tdf['search_term']]

# query feature
class AmountOfNumericalCharactersInQuery:
    def extract(self, tdf, tdf_un, ndf):
        ndf['amount_numericals'] = [sum(s.isdigit() for s in x.split()) for x in tdf['search_term']]

# query feature
class AmountOfNumbersInQuery:
    def extract(self, tdf, tdf_un, ndf):
        ndf['amount_numbers'] = [sum(s.isdigit() for s in nltk.word_tokenize((x))) for x in tdf['search_term']]



## ratios, percentages etc.

# query feature
class PercOfQueryCharsNumerical:
    def extract(self, tdf, tdf_un, ndf):
        query_length_in_chars = [len(x) for x in tdf['search_term']]
        amount_numericals = [sum(s.isdigit() for s in x.split()) for x in tdf['search_term']]
        ndf['perc_of_query_numerical'] = [y/x for x,y in zip(query_length_in_chars, amount_numericals)]

# query feature
class PercOfQueryTokensNumerical:
    def extract(self, tdf, tdf_un, ndf):
        query_length_in_tokens = [len(nltk.word_tokenize(x)) for x in tdf['search_term']]
        amount_numbers = [sum(s.isdigit() for s in nltk.word_tokenize((x))) for x in tdf['search_term']]
        ndf['perc_of_query_number'] = [y/x for x,y in zip(query_length_in_tokens, amount_numbers)]

# percentage of special characters in query
class PercOfQueryCharsOther:
    def extract(self, tdf, tdf_un, ndf):
        query_length_in_chars = [len(x) for x in tdf['search_term']]
        query_others = [countothers(x) for x in tdf['search_term']]
        ndf['perc_of_query_nonspacenonalphanonnum_chars'] = [(x)/(y) for x,y in zip(query_others, query_length_in_chars)]

# percentage of alphabetical characters in query
class PercOfQueryCharsAlphabetical:
    def extract(self, tdf, tdf_un, ndf):
        query_length_in_chars = [len(x) for x in tdf['search_term']]
        query_alphas = [countchars(x) for x in tdf['search_term']]
        ndf['perc_of_query_alphabetical_chars'] = [(x)/(y) for x,y in zip(query_alphas, query_length_in_chars)]

# percentage of spaces in query
class PercOfQueryCharsSpaces:
    def extract(self, tdf, tdf_un, ndf):
        query_length_in_chars = [len(x) for x in tdf['search_term']]
        query_spaces = [countspaces(x) for x in tdf['search_term']]
        ndf['perc_of_query_spaces_chars'] = [(x)/(y) for x,y in zip(query_spaces, query_length_in_chars)]

# query feature
# ratio of alphabetical vs numerical characters
class RatioAlphaVsNumInQueryChars:

    def extract(self, tdf, tdf_un, ndf):
        ndf['query_alpha_num_ratio_chars'] = [alpha_num_ratio(x) for x in tdf['search_term']]

# # query feature
# # ratio of alphabetical vs numerical characters
# class RatioAlphaVsNumInQueryChars:

#     def extract(self, tdf, tdf_un, ndf):

#         numericals = ndf['count_of_num_chars']
#         if numericals == 0:
#             ndf['query_alpha_num_ratio_chars'] = 0
#         else:
#             alpha_num_ratio = (alphas)/(numericals)
#             ndf['query_alpha_num_ratio_chars'] = [(alphas)/(numericals) for alphas,numericals in zip(ndf['count_of_alph_chars'],ndf['count_of_num_chars'])]


# query feature
# ratio of only-alphabetical vs only-numerical tokens
class RatioAlphaVsNumInQueryTokens:

    def extract(self, tdf, tdf_un, ndf):
        ndf['query_alpha_num_ratio_tokens'] = [alpha_num_ratio(nltk.word_tokenize((x))) for x in tdf['search_term']]

# ratio of only-alphabetical vs mixed?
# ratio of only-numerical vs mixed?

# query feature
# ratio of alphabetical characters vs spaces
class RatioAlphasVsSpacesInQuery:

    def alpha_space_ratio(self, x):
        numericals, alphas, spaces, others = charcount(x)
        if spaces == 0:
            return 0
        else:
            alpha_space_ratio_var = (alphas)/(spaces)
            return alpha_space_ratio_var

    def extract(self, tdf, tdf_un, ndf):
        ndf['query_alpha_space_ratio'] = [self.alpha_space_ratio("".join(x)) for x in tdf['search_term']]

# query feature
# ratio of numerical characters vs spaces
class RatioNumericalsVsSpacesInQuery:

    def num_spaces_ratio(self, x):
        numericals, alphas, spaces, others = charcount(x)
        if spaces == 0:
            return 0
        else:
            num_spaces_ratio = (numericals)/(spaces)
            return num_spaces_ratio

    def extract(self, tdf, tdf_un, ndf):
        ndf['query_num_space_ratio'] = [self.num_spaces_ratio("".join(x)) for x in tdf['search_term']]

# query feature
# ratio of words (purely alphabetical tokens) vs non-words (non-purely alphabetical tokens)
class RatioWordsVsNonwords:


    def ElementHasNumbers(self, tokens):
        nonwords = []
        nonwords = [1 if bool(re.search(r'\d', token)) else 0 for token in tokens]
        return nonwords

    def words_nonwords(self, x):
        tokens = []
        tokens = x.split(" ")
        nonwords = sum(self.ElementHasNumbers(tokens))
        words = len(tokens) - nonwords
        return words, nonwords

    def words_nonwords_ratio(self, x):
        words, nonwords = self.words_nonwords(x)
        ratio = (words + 1)/(nonwords + 1)
        return ratio
    

    def extract(self, tdf, tdf_un, ndf):
        ndf['query_words_nonwords_ratio'] = [self.words_nonwords_ratio(x) for x in tdf['search_term']]












################
###### MISC
################

## misc (still query features)

# was spelling-correction performed?
class SpellingCorrectionPerformed:
    def extract(self, tdf, tdf_un, ndf):
        ndf['spell_corrected'] = [x for x in tdf['spell_corrected']]

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
        

class TermFrequency:
    def extract(self, tdf, tdf_un, ndf):
        with open('data/termcounts/dict.pkl', 'rb') as fcounts:
            counts = pickle.load(fcounts)

            ndf['mean_query_tf'] = [np.mean([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]
            ndf['min_query_tf'] = [np.min([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]
            ndf['max_query_tf'] = [np.max([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]
            ndf['std_query_tf'] = [np.std([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]

class TermFrequencyProductTitle:
    def extract(self, tdf, tdf_un, ndf):
        with open('data/termcounts/dict.pkl', 'rb') as fcounts:
            counts = pickle.load(fcounts)

            ndf['mean_title_tf'] = [np.mean([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]
            ndf['min_title_tf'] = [np.min([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]
            ndf['max_title_tf'] = [np.max([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]
            ndf['std_title_tf'] = [np.std([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]


class DocFrequencyQueryQuery:
    def extract(self, tdf, tdf_un, ndf):
        with open('data/termcounts/doccounts.pkl', 'rb') as fcounts:
            counts = pickle.load(fcounts)

            ndf['mean_query_df'] = [np.mean([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                    tdf['search_term']]
            ndf['min_query_df'] = [np.min([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]
            ndf['max_query_df'] = [np.max([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]
            ndf['std_query_df'] = [np.std([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]

class DocFrequencyTitleQuery:
    def extract(self, tdf, tdf_un, ndf):
        with open('data/termcounts/doccounts.pkl', 'rb') as fcounts:
            counts = pickle.load(fcounts)

            ndf['mean_query_title_df'] = [np.mean([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                    tdf['product_title']]
            ndf['min_query_title_df'] = [np.min([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]
            ndf['max_query_title_df'] = [np.max([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]
            ndf['std_query_title_df'] = [np.std([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]

class DocFrequencyQueryTitle:
    def extract(self, tdf, tdf_un, ndf):
        with open('data/termcounts/prodcounts.pkl', 'rb') as fcounts:
            counts = pickle.load(fcounts)

            ndf['mean_title_query_df'] = [np.mean([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                    tdf['search_term']]
            ndf['min_title_query_df'] = [np.min([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]
            ndf['max_title_query_df'] = [np.max([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]
            ndf['std_query_query_df'] = [np.std([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['search_term']]

class DocFrequencyTitleTitle:
    def extract(self, tdf, tdf_un, ndf):
        with open('data/termcounts/prodcounts.pkl', 'rb') as fcounts:
            counts = pickle.load(fcounts)

            ndf['mean_title_df'] = [np.mean([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                    tdf['product_title']]
            ndf['min_title_df'] = [np.min([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]
            ndf['max_title_df'] = [np.max([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]
            ndf['std_title_df'] = [np.std([counts[y] if y in counts else 0 for y in x.split()]) for x in
                                   tdf['product_title']]

class DescriptionLength:
    def extract(self, tdf, tdf_un, ndf):
        ndf['descr_length'] = [len(x.split()) for x in tdf['product_description']]
        ndf['descr_length_chars'] = [len(x) for x in tdf['product_description']]
        ndf['descr_av_word_length'] = [y / x for x, y in zip(ndf['descr_length'], ndf['descr_length_chars'])]

class TitleLength:
    def extract(self, tdf, tdf_un, ndf):
        ndf['title_length'] = [len(x.split()) for x in tdf['product_title']]
        ndf['title_length_chars'] = [len(x) for x in tdf['product_title']]
        ndf['title_av_word_length'] = [y / x for x, y in zip(ndf['title_length'], ndf['title_length_chars'])]

class RelativeLengths:
    def extract(self, tdf, tdf_un, ndf):
        querylen = [len(x.split()) for x in tdf['search_term']]
        titlelen = [len(x.split()) for x in tdf['product_title']]
        descrlen = [len(x.split()) for x in tdf['product_description']]

        ndf['rel_query_title_length'] = [y / x for x, y in zip(querylen, titlelen)]
        ndf['rel_query_descr_length'] = [y / x for x, y in zip(querylen, descrlen)]
        ndf['rel_title_descr_length'] = [y / x for x, y in zip(titlelen, descrlen)]





######
## BINARY FEATURES
######
