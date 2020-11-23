import pandas as pd
import numpy as np
import scipy
import fasttext

'''
new plan:
use this file as a thing that can set up the model
so it finishes up cleaning, formats everything correctly 
for training.
Than make a new file that tests the model on its precision 
and recall with the test set.


reviews_training.json
to a dataframe
clean
save as a csv reviews_training.csv
read in csv as textfile
reformat
save as a text file reviews.training.txt
'''



print("*")
df = pd.read_json(r'reviews_train.json')

test = df.head()
print(test)

# PRE PROCESSING
# encode the cleaned text
print('------')

words = test.filter(['Id','Score', 'clean'],axis=1)
print(words)
print("----------")
# encode UFT-8
words['clean'] = words['clean'].apply(lambda w : w.encode())
print(words)

# output as text file with proper format
# first output as csv, then read back in as txt
df.to_csv(r'')

# format: 	__label__<X>__label__<Y> ... <Text>
np.savetxt(r'cleaned_train.txt', words.to_numpy.values)


# model = fasttext.train_supervised(

'''
d = fasttext.tokenize("light pillowy citrus")
print("_")
print(d)
'''