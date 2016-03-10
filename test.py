__author__ = 'martijn'
from nltk.stem.porter import *


pattern = '.*(Width|Height|Depth|Length).*'

m = re.match(pattern, "Product Height (in.)")
if(m):
    print("succes")