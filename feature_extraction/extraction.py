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
            'Average position of matched query words': N.AveragePositionMatchedSearchTerms(),
            'Title Overlap': N.TitleOverlap(),
            'Word2Vec Similarity': N.Word2VecSimilarity(),
            'Word2Vec Summed Similarity': N.Word2vecSummedSimilarity(),
            'Word2Vec Similarity Pretrained': N.Word2VecSimilarityPretrained(),
            'Word2Vec Summed Similarity Pretrained': N.Word2VecSummedSimilarityPretrained(),
            'Description Overlap': N.DescriptionOverlap(),
            'Description Overlap Jaccard': N.DescriptionOverlapJaccard(),
            'Description Match': N.DescriptionMatch(),
            'Title 3-gram Overlap': N.TitleOverlapNgram(),
            'Title 3-gram Overlap Jaccard': N.TitleOverlapNgramJaccard(),
            'Title 3-gram Match': N.TitleMatchNgram(),
            'Brand Match': N.BrandMatch(),
            'Color Overlap': N.ColorOverlap(),
            'Color Match': N.ColorMatch(),
            'Number of Words': N.QueryLength(),
            'Number of 3-grams': N.QueryLengthNgram(),
            'Number of Characters': N.QueryCharachterLength(),
            'Average length of word': N.QueryAverageWordLength(),
            'Ratio of 3-grams matching in Title': N.RatioNgramsInQueryMatchInTitle(),
            'Amount of Numbers': N.AmountOfNumbersInQuery(),
            'Ratio of Numbers': N.RatioNumbersInQuery(),
            'Number of Vowels in Search Term': N.NumberOfVowelsSearchTerm(),
            'Number of Nouns': N.NumberOfNouns(),
            'Spelling Correction Performed': N.SpellingCorrectionPerformed(),
            'Number of Vowels in Title': N.NumberOfVowelsTitle(),
            'Distance between title matched terms': N.DistanceMatchedSearchTerms(),
            'Average Term Frequency of Query': N.AverageTermFrequency(),
            'Minimum Term Frequency of Query': N.MinimumTermFrequency(),
            'Maximum Term Frequency of Query': N.MaximumTermFrequency(),
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
            df.to_csv('data/features/text_{}.csv'.format(self.name), index=False)

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
            ndf.to_csv('data/features/numerical_{}.csv'.format(self.name), index=False)

        return ndf