import pandas as pd
import numpy
import scipy
import fasttext

df = pd.read_json(r'Reviews.json')

test = df.head()
print(test)

# encode the cleaned text
print('------')

words = test.filter(['Id','Score', 'clean'],axis=1)
print(words)
print("----------")
words['clean'] = words['clean'].apply(lambda w : w.encode())
print(words)
'''
d = fasttext.tokenize("light pillowy citrus")
print("_")
print(d)
'''