import pandas as pd
import feature_extraction.textual as T
import feature_extraction.numerical as N

class FeatureExtractor:

    def __init__(self, descriptionDF, attributeDF, verbose=False):
        self.isVerbose = verbose
        self.textualExtractors = {
            'Search Term Ngrams': T.SearchTermNgrams(),
            'Product Title Ngrams': T.ProductTitleNgrams(),
            'Brand Names': T.BrandNames(attributeDF),
            'Product Description': T.ProductDescription(descriptionDF),
            'Colors': T.Colors(attributeDF)
        }
        self.numericalExtractors = {
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
            'Number of Nouns': N.NumberOfNouns(),
        }

    def extractTextualFeatures(self, df):
        for key, extractor in self.textualExtractors.items():
            if (self.isVerbose):
                print("Textual feature extraction: {}".format(key))
            extractor.extract(df)

        return df

    def extractNumericalFeatures(self, df):
        ndf = pd.DataFrame()
        for key, extractor in self.numericalExtractors.items():
            if (self.isVerbose):
                print("Numerical feature extraction: {}".format(key))
            extractor.extract(df, ndf)

        return ndf