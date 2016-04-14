## feature_extraction/extraction.py

import pandas as pd
import feature_extraction.textual as T
import feature_extraction.numerical as N
import os

class FeatureExtractor:

    def __init__(self, descriptionDF, attributeDF, verbose=False, name=""):
        self.isVerbose = verbose
        self.name = name
        self.textualExtractors = {
            'Search Term Ngrams': T.SearchTermNgrams(),
            'Product Title Ngrams': T.ProductTitleNgrams(),
            'Brand Names': T.BrandNames(attributeDF),
            'Product Description': T.ProductDescription(descriptionDF),
            'Colors': T.Colors(attributeDF)
        }
        self.numericalExtractors = {
<<<<<<< HEAD
=======
            'Word2Vec Similarity': N.Word2VecSimilarity(),
            'Word2Vec Similarity Pretrained': N.Word2VecSimilarityPretrained(),
>>>>>>> 92f2492e1a532bcdaf1db7663bc32a5e21b5b502
            'Description Overlap': N.DescriptionOverlap(),
            'Description Overlap Jaccard': N.DescriptionOverlapJaccard(),
            'Description Match': N.DescriptionMatch(),
            'Title 3-gram Overlap': N.TitleOverlapNgram(),
            'Title 3-gram Overlap Jaccard': N.TitleOverlapNgramJaccard(),
            'Title 3-gram Match': N.TitleMatchNgram(),
            'Brand Match': N.BrandMatch(),
            'Color Overlap': N.ColorOverlap(),
            'Color Match': N.ColorMatch(),
            'Query Length by Tokens': N.QueryLengthByTokens(),
            'Query Length by 3-grams': N.QueryLengthByNgrams(),
            'Query Length by Characters': N.QueryLengthByCharachter(),
            'Average length of word': N.QueryAverageTokenLength(),
            'Ratio of 3-grams matching in Title': N.RatioNgramsInQueryMatchInTitle(),
            'Amount of Numbers': N.AmountOfNumbersInQuery(),
            'Ratio of Numbers': N.RatioNumbersInQuery(),
<<<<<<< HEAD
            'Spelling Correction Performed': N.SpellingCorrectionPerformed(),
            'Word2Vec Similarity of Query and PTitle': N.Word2VecSimilarity(),
            'Query: Counts of Characters per Class': N.CountsOfCharsPerClass(),
            'Query: Counts of Tokens per Class': N.CountsOfTokensPerClass(),
            'Ratio of alphabeticals to numericals in query (in terms of characters)': N.RatioAlphaVsNumInQueryChars(),
            'Ratio of alphabeticals to numericals in query (in terms of tokens)': N.RatioAlphaVsNumInQueryTokens(),
            'Ratio of alphabeticals to spaces in query': N.RatioAlphasVsSpacesInQuery(),
            'Ratio of numericals to spaces in query': N.RatioNumericalsVsSpacesInQuery(),
            'Ratio of words to nonwords in query': N.RatioWordsVsNonwords(),
            'Count of special (non-alphabetical, non-numerical, non-space) chars in Q': N.LengthNonSpaceNonAlphaNonNumericalChars(),
            'Count of non-purely alpha/numerical/space tokens in Q': N.LengthNonSpaceNonAlphaNonNumericalTokens(),
            'Count of Nouns in Query': N.NumberOfNouns(),
=======
            'Number of Vowels in Search Term': N.NumberOfVowelsSearchTerm(),
            'Number of Nouns': N.NumberOfNouns(),
            'Spelling Correction Performed': N.SpellingCorrectionPerformed(),
            'Number of Vowels in Title': N.NumberOfVowelsTitle(),
>>>>>>> 92f2492e1a532bcdaf1db7663bc32a5e21b5b502
        }

    def extractTextualFeatures(self, df, saveResults=False):
        if saveResults and os.path.isfile('data/features/text_{}.csv'.format(self.name)):
            if (self.isVerbose):
                print("Textual feature extraction: reading from saved file")
            return pd.read_csv('data/features/text_{}.csv'.format(self.name))

        for key, extractor in self.textualExtractors.items():
            if (self.isVerbose):
                print("Textual feature extraction: {}".format(key))
            extractor.extract(df)

        if saveResults:
            df.to_csv('data/features/text_{}.csv'.format(self.name))

        return df

    def extractNumericalFeatures(self, df, df_un, saveResults=False):
        if saveResults and os.path.isfile('data/features/numerical_{}.csv'.format(self.name)):
            if (self.isVerbose):
                print("Numerical feature extraction: reading from saved file")
            return pd.read_csv('data/features/numerical_{}.csv'.format(self.name))

        ndf = pd.DataFrame()
        for key, extractor in self.numericalExtractors.items():
            if (self.isVerbose):
                print("Numerical feature extraction: {}".format(key))
            extractor.extract(df, df_un, ndf)

        if saveResults:
            ndf.to_csv('data/features/numerical_{}.csv'.format(self.name))

        return ndf