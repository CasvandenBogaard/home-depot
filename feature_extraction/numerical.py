## this is feature_extraction/numerical.py

import math
import nltk
import os
import gensim.models.word2vec as w2v
import re


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




# query feature
class QueryLengthByTokens:
    def extract(self, tdf, ndf):
        ndf['query_token_length'] = [len(x.split()) for x in tdf['search_term']]

# query feature
class QueryLengthByNgrams:
    def extract(self, tdf, ndf):
        ndf['query_length_ngram'] = [len(x.split()) for x in tdf['search_term_ngram']]

# query feature
class QueryLengthByCharachters:
    def extract(self, tdf, ndf):
        ndf['query_character_length'] = [len(x) for x in tdf['search_term']]

# query feature
class QueryAverageTokenLength:
    def extract(self, tdf, ndf):
        query_length = [len(x.split()) for x in tdf['search_term']]
        query_char_length = ndf['query_character_length'] = [len(x) for x in tdf['search_term']]
        ndf['query_average_token_length'] = [y/x for x,y in zip(query_length, query_char_length)]

# query feature
class RatioNgramsInQueryMatchInTitle:
    def extract(self, tdf, ndf):
        query_ngram_length = [len(x.split()) for x in tdf['search_term_ngram']]
        title_ngram_overlap = [sum(int(word in y) for word in x.split()) for x,y in zip(tdf['search_term_ngram'], tdf['product_title_ngram'])]
        ndf['total_match_title'] = [math.floor(x/y) for x,y in zip(title_ngram_overlap, query_ngram_length)]

# query feature
class AmountOfNumbersInQuery:
    def extract(self, tdf, ndf):
        ndf['amount_numbers'] = [sum(s.isdigit() for s in x.split()) for x in tdf['search_term']]

# query feature
class RatioNumbersInQuery:
    def extract(self, tdf, ndf):
        query_char_length = [len(x) for x in tdf['search_term']]
        amount_numbers = [sum(s.isdigit() for s in x.split()) for x in tdf['search_term']]
        ndf['ratio_numbers'] = [y/x for x,y in zip(query_char_length, amount_numbers)]

# was spelling-correction performed?
class SpellingCorrectionPerformed:
    def extract(self, tdf, ndf):
        ndf['spell_corrected'] = [x for x in tdf['spell_corrected']]

# word2vec similarity of query and product title
class Word2VecSimilarity:
    def extract(self, tdf, ndf):
        if os.path.isfile('data/word2vec/full'):
            model = w2v.Word2Vec.load('data/word2vec/full')
            ndf['word2vec_sim'] = [
                sum(
                    model.similarity(q, t)
                    for q in x.split() if q in model.vocab
                    for t in y.split() if t in model.vocab
                )
                for x,y in zip(tdf['search_term'], tdf['product_title'])
            ]



##
# these functions are used in the classes below
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

# query feature
## absolute occurrence of character class (numerical, alphabetical, spaces, others (signs etc.))
class CountsOfCharsPerClass:
    def extract(self, tdf, ndf):
        ndf['count_of_num_chars'] = [countdigits(x) for x in tdf['search_term']]
        ndf['count_of_alph_chars'] = [countchars(x) for x in tdf['search_term']]
        ndf['count_of_space_chars'] = [countspaces(x) for x in tdf['search_term']]
        ndf['count_of_other_chars'] = [countothers(x) for x in tdf['search_term']]

# query feature
## absolute occurrence of token class (numerical only, alphabethical only, mixed only)
class CountsOfTokensPerClass:
    def extract(self, tdf, ndf):
        ndf['count_of_pure_num_tokens'] = [countdigits(nltk.word_tokenize((x))) for x in tdf['search_term']]
        ndf['count_of_pure_alph_tokens'] = [countchars(nltk.word_tokenize((x))) for x in tdf['search_term']]
        ndf['count_of_other_tokens'] = [countothers(nltk.word_tokenize((x))) for x in tdf['search_term']]


## ratios!

# helper function
def alpha_num_ratio(x):
    numericals, alphas, spaces, others = charcount(x)
    if numericals == 0:
        return 0
    else:
        alpha_num_ratio = (alphas)/(numericals)
        return alpha_num_ratio

# query feature
# ratio of alphabetical vs numerical characters
class RatioAlphaVsNumInQueryChars:

    def extract(self, tdf, ndf):
        ndf['query_alpha_num_ratio_chars'] = [alpha_num_ratio(x) for x in tdf['search_term']]


# # query feature
# # ratio of alphabetical vs numerical characters
# class RatioAlphaVsNumInQueryChars:

#     def extract(self, tdf, ndf):

#         numericals = ndf['count_of_num_chars']
#         if numericals == 0:
#             ndf['query_alpha_num_ratio_chars'] = 0
#         else:
#             alpha_num_ratio = (alphas)/(numericals)
#             ndf['query_alpha_num_ratio_chars'] = [(alphas)/(numericals) for alphas,numericals in zip(ndf['count_of_alph_chars'],ndf['count_of_num_chars'])]

# query feature
# ratio of only-alphabetical vs only-numerical tokens
class RatioAlphaVsNumInQueryTokens:

    def extract(self, tdf, ndf):
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

    def extract(self, tdf, ndf):
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

    def extract(self, tdf, ndf):
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
    
    def extract(self, tdf, ndf):
        ndf['query_words_nonwords_ratio'] = [self.words_nonwords_ratio(x) for x in tdf['search_term']]



## more length!

# query feature
# absolute number of special characters in query
class LengthNonSpaceNonAlphaNonNumericalChars:

    def count_nonspacenonalphanonnumerical(self, x):
        numericals, alphas, spaces, others = charcount(x)
        return others

    def extract(self, tdf, ndf):
        ndf['query_length_nonspacenonalphanonnum_chars'] = [self.nonspacenonalphanonnumerical(x) for x in tdf['search_term']]

# query feature
# absolute number of mixed (not purely-alphabetical or purely-numerical) tokens in query
class LengthNonSpaceNonAlphaNonNumericalTokens:

    def count_nonspacenonalphanonnumerical(self, x):
        numericals, alphas, spaces, others = charcount(x)
        return others

    def extract(self, tdf, ndf):
        ndf['query_length_nonspacenonalphanonnum_tokens'] = [self.nonspacenonalphanonnumerical(nltk.word_tokenize((x))) for x in tdf['search_term']]

# query feature
# number of nouns in query by nltk POS-tags
class NumberOfNouns:
    def extract(self, tdf, ndf):
        sentencelist = [x.split() for x in tdf['search_term']]
        result = nltk.pos_tag_sents(sentencelist)
        nouns = [[word for word,pos in lst if pos in ['NN', 'NNP', 'NNS', 'NNPS']] for lst in result]

        ndf['number_of_nouns'] = [int(len(x)) for x in nouns]

# query feature
# number of adjectives in query? etc etc (other POS-tags)