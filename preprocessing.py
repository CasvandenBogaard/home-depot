import pandas as pd

from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

def str_stem(s):
    if isinstance(s, str):
        s = s.lower()
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)

        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)

        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)

        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)

        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")

        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)

        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)

        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)

        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)

        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)

        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)

        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)

        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)

        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)

        s = s.replace("&amp;", "and")
        s = s.replace("&", "and")

        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

        s = s.replace("  "," ")
        s = (" ").join([stemmer.stem(z) for z in s.lower().split(" ")])
        return s.lower()
    else:
        return "null"

stemmer = SnowballStemmer('english')

df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
df_description = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
df_attributes = pd.read_csv('data/attributes.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")

# print("stemming training product title")
# df_train["product_title"] = df_train["product_title"].map(lambda x:str_stem(x))
# print("stemming training search term")
# df_train["search_term"] = df_train["search_term"].map(lambda x:str_stem(x))
# print("stemming product description")
# df_description["product_description"] = df_description["product_description"].map(lambda x:str_stem(x))
# print("stemming value")
# df_attributes["value"] = df_attributes["value"].map(lambda x: str_stem(x))

print("stemming test product title")
df_test["product_title"] = df_test["product_title"].map(lambda x:str_stem(x))
print("stemming test search term")
df_test["search_term"] = df_test["search_term"].map(lambda x:str_stem(x))

#df_train.to_csv('data/stemmed/train.csv', index=False)
#df_description.to_csv('data/stemmed/product_descriptions.csv', index=False)
#df_attributes.to_csv('data/stemmed/attributes.csv', index=False)
df_test.to_csv('data/stemmed/test.csv', index=False)