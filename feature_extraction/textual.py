import pandas as pd

class ProductDescription:
    def __init__(self, descriptionDF):
        self.descriptionDF = descriptionDF

    def extract(self, df):
        tempdf = pd.merge(df, self.descriptionDF, how='left', on='product_uid')
        df['product_description'] = tempdf['product_description']

class BrandNames:
    def __init__(self, attributesDF):
        self.attributesDF = attributesDF

    def extract(self, df):
        brands = self.attributesDF[self.attributesDF.name=='MFG Brand Name']
        tempdf = pd.merge(df, brands, how='left', on='product_uid')
        df['brand'] = tempdf['value']

class Colors:
    def __init__(self, attributesDF):
        self.attributesDF = attributesDF

    def extract(self, df):
        color_attribute_mask = ['Color' in str(name) for name in self.attributesDF['name']]
        colors = self.attributesDF.loc[color_attribute_mask,:]
        df_atrr = colors.drop('name', axis=1)
        df_atrr["value"] = df_atrr["value"].astype(str)
        grouped = df_atrr.groupby('product_uid').apply(lambda x: (" ").join(x.value))
        groupeddf = grouped.reset_index()
        groupeddf.columns = ['product_uid', 'colors']
        tempd = pd.merge(df, groupeddf, how='left', on='product_uid')

        df['colors'] = tempd['colors']

class SearchTermNgrams:
    def n_gram(self, attribute, data, n_gram_attr, n):
        data[str(n_gram_attr)] = data[str(attribute)]

        ngrams = lambda b, n: [b[i:i+n] for i in range(len(b)-n+1)]
        wordlist = [x.split() for x in data[str(attribute)]]

        ng = [[ngrams(y,n) if len(y) >= n else [y] for y in x] for x in wordlist]
        result = [[item for sublist in x for item in sublist] for x in ng]
        data[str(n_gram_attr)] = [" ".join(x) for x in result]

    def extract(self, df):
        self.n_gram('search_term', df, 'search_term_ngram', 3)

class ProductTitleNgrams:
    def n_gram(self, attribute, data, n_gram_attr, n):
        data[str(n_gram_attr)] = data[str(attribute)]

        ngrams = lambda b, n: [b[i:i+n] for i in range(len(b)-n+1)]
        wordlist = [x.split() for x in data[str(attribute)]]

        ng = [[ngrams(y,n) if len(y) >= n else [y] for y in x] for x in wordlist]
        result = [[item for sublist in x for item in sublist] for x in ng]
        data[str(n_gram_attr)] = [" ".join(x) for x in result]

    def extract(self, df):
        self.n_gram('product_title', df, 'product_title_ngram', 3)
