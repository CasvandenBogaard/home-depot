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
            'Search Term 2grams': T.SearchTerm2grams(),
            'Search Term 3grams': T.SearchTerm3grams(),
            'Search Term 4grams': T.SearchTerm4grams(),
            'Product Title 2grams': T.ProductTitle2grams(),
            'Product Title 3grams': T.ProductTitle3grams(),
            'Product Title 4grams': T.ProductTitle4grams(),
            'Brand Names': T.BrandNames(attributeDF),
            'Product Description': T.ProductDescription(descriptionDF),
            'Colors': T.Colors(attributeDF)
        }
        self.numericalExtractors = {
            'Average position of matched query words': N.AveragePositionMatchedSearchTerms(),
            'Title Overlap': N.TitleOverlap(),
            'Word2Vec Similarity of Query and PTitle': N.Word2VecSimilarity(),
            'Word2Vec Similarity Pretrained of Query and PTitle': N.Word2VecSimilarityPretrained(),
            'Word2Vec Summed Similarity': N.Word2vecSummedSimilarity(),
            'Word2Vec Summed Similarity Pretrained': N.Word2VecSummedSimilarityPretrained(),
            'Description Overlap': N.DescriptionOverlap(),
            'Description Overlap Jaccard': N.DescriptionOverlapJaccard(),
            'Description Match': N.DescriptionMatch(),
            'Title 2-gram Overlap': N.TitleOverlap2gram(),
            'Title 2-gram Overlap Jaccard': N.TitleOverlap2gramJaccard(),
            'Title 2-gram Match': N.TitleMatch2gram(),
            'Title 3-gram Overlap': N.TitleOverlap3gram(),
            'Title 3-gram Overlap Jaccard': N.TitleOverlap3gramJaccard(),
            'Title 3-gram Match': N.TitleMatch3gram(),
            'Title 4-gram Overlap': N.TitleOverlap4gram(),
            'Title 4-gram Overlap Jaccard': N.TitleOverlap4gramJaccard(),
            'Title 4-gram Match': N.TitleMatch4gram(),
            'Brand Match': N.BrandMatch(),
            'Color Overlap': N.ColorOverlap(),
            'Color Match': N.ColorMatch(),
            'Number of Words': N.QueryLengthByTokens(),
            'Number of 2-grams': N.QueryLength2gram(),
            'Number of 3-grams': N.QueryLength3gram(),
            'Number of 4-grams': N.QueryLength4gram(),
            'Query Length by Tokens': N.QueryLengthByTokens(),            
            'Query Length by Characters': N.QueryLengthByCharacters(),
            'Average length of word': N.QueryAverageTokenLength(),
            'Ratio of 2-grams matching in Title': N.Ratio2gramsInQueryMatchInTitle(),
            'Ratio of 3-grams matching in Title': N.Ratio3gramsInQueryMatchInTitle(),
            'Ratio of 4-grams matching in Title': N.Ratio4gramsInQueryMatchInTitle(),
            'Amount of Numbers': N.AmountOfNumbersInQuery(),
            'Amount of Numericals': N.AmountOfNumericalCharactersInQuery(),
            'Percent of Query Tokens numerical': N.PercOfQueryTokensNumerical(),
            'Percent of Query characters numerical': N.PercOfQueryCharsNumerical(),
            'Percent of Query characters Others': N.PercOfQueryCharsOther(),
            'Percent of Query characters alphabetical': N.PercOfQueryCharsAlphabetical(),
            'Percent of Query characters spaces': N.PercOfQueryCharsSpaces(),
            'Percent of Query Characters special': N.PercOfQueryCharsOther(),
            'Spelling Correction Performed': N.SpellingCorrectionPerformed(),
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
            'Number of Vowels in Search Term': N.NumberOfVowelsSearchTerm(),
            # missing in numerical.py? 'Number of Vowels in Title': N.NumberOfVowelsTitle(),
            'Distance between title matched terms': N.DistanceMatchedSearchTerms(),
            'Last word in query title':N.LastWordInTitle(),
            'First word in query title':N.FirstWordInTitle(),
            'Avg, Min, Max, Std Term Frequency of Query': N.TermFrequency(),
            'Avg, Min, Max, Std Term Frequency of Product Title': N.TermFrequencyProductTitle(),
            'Avg, Min, Max, Std Query Frequency of Query words': N.DocFrequencyQueryQuery(),
            'Avg, Min, Max, Std Query Frequency of Product Title words': N.DocFrequencyTitleQuery(),
            'Avg, Min, Max, Std Product Frequency of Query words': N.DocFrequencyQueryTitle(),
            'Avg, Min, Max, Std Product Frequency of Product Title words': N.DocFrequencyTitleTitle(),
            'Description Lengths': N.DescriptionLength(),
            'Product Title Lengths': N.TitleLength(),
            'Relative Lengths': N.RelativeLengths(),
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
            ndf = pd.read_csv('data/features/numerical_{}.csv'.format(self.name))

            for key in self.getFeaturesToRefresh():
                extractor = self.numericalExtractors[key]
                if (self.isVerbose):
                    print("Numerical feature refreshing: {}".format(key))
                extractor.extract(df, df_un, ndf)

            ndf.to_csv('data/features/numerical_{}.csv'.format(self.name), index=False)
            return ndf

        ndf = pd.DataFrame()
        for key, extractor in self.numericalExtractors.items():
            if (self.isVerbose):
                print("Numerical feature extraction: {}".format(key))
            extractor.extract(df, df_un, ndf)

        if saveResults:
            ndf.to_csv('data/features/numerical_{}.csv'.format(self.name))

        return ndf

    def getFeaturesToRefresh(_self):
        return [
            # "Average Term Frequency of Query",
            # "Minimum Term Frequency of Query",
            # "Maximum Term Frequency of Query",
        ]