import pandas as pd
import numpy as np
import scipy
import fasttext

'''
Run this file to make text files fromatted for training FastText models,
the trained models themselves (saved in /models), and run a series of
tests that compare the accuracy of the different models.
'''

def format_uncleaned(df) :
    train = df.filter(['Score','Text'], axis=1)
    train['Score'] =train['Score'].apply(lambda r : "__label__" + str(r))

    train.to_csv(r'reviews_unclean_train.csv')

    # format: 	__label__<X>__label__<Y> ... <Text>
    # encoding must be UTF-8
    with open('reviews_unclean_train.csv', mode='r', encoding='UTF-8') as f :
        train = f.read()
        lines = train.split('\n')
        lines.remove(lines[0])
        lines.remove(lines[-1]) 
     
        formatted_lines = []
        for line in lines :
            i = line.find(',')
            line = line[i+1:]
            
            # Assume that commas can appear in text field 
            # i.e. don't use sr.replace()
            # This is necessary for the raw data.
            j = line.find(',')
            line = list(line)
            line[j] = ' '
            line = ''.join(line)
            formatted_lines.append(line)

        train = '\n'.join(formatted_lines)
        print("Preprocessed uncleaned data:")
        print(train[:500])

        final = open("reviews_uncleaned.train", mode='w+', encoding='UTF-8')
        final.write(train)




def format(df) :
    print()
    print('Preprocessing the data:')

    train = df.filter(['Score', 'clean'],axis=1)

    # Star ratings are labels with format "__<label>__#Score"
    train['Score'] =train['Score'].apply(lambda r : "__label__" + str(r))

    # Output as text file with proper format
    # First output as csv, then read back in as txt because
    # the csv's plaintext is easier to manipulate
    train.to_csv(r'reviews_train.csv')

    # format: 	__label__<X>__label__<Y> ... <Text>
    # encoding must be UTF-8
    with open('reviews_train.csv', mode='r', encoding='UTF-8') as f :
        train = f.read()
        lines = train.split('\n')
        lines.remove(lines[0])
        lines.remove(lines[-1]) 
     
        formatted_lines = []
        for line in lines :
            i = line.find(',')
            line = line[i+1:]
            
            # Assume that commas can appear in text field 
            # i.e. don't use sr.replace()
            # This is necessary for the raw data.
            j = line.find(',')
            line = list(line)
            line[j] = ' '
            line = ''.join(line)
            formatted_lines.append(line)

        train = '\n'.join(formatted_lines)
        print("Preprocessed cleaned data:")
        print(train[:500])

        final = open("reviews_cleaned.train", mode='w+', encoding='UTF-8')
        final.write(train)



def test() :
    # ///////// Build cleaned test set \\\\\\\\\\
    df = pd.read_json(r'reviews_test.json')
    test = df.filter(['Score', 'clean'],axis=1)
    test['Score'] = test['Score'].apply(lambda r : "__label__" + str(r))
    test.to_csv(r'reviews_test.csv')

    # format: 	__label__<X>__label__<Y> ... <Text>
    with open('reviews_test.csv', mode='r', encoding='UTF-8') as f :
        test = f.read()
        lines = test.split('\n')
        lines.remove(lines[0])
        lines.remove(lines[-1]) 
     
        formatted_lines = []
        for line in lines :
            i = line.find(',')
            line = line[i+1:]
            
            # Assume that commas can appear in text field 
            j = line.find(',')
            line = list(line)
            line[j] = ' '
            line = ''.join(line)
            formatted_lines.append(line)

        test = '\n'.join(formatted_lines)
        final = open("reviews_cleaned.test", mode='w+', encoding='UTF-8')
        final.write(test)
    
    # //////// Build uncleaned test set \\\\\\\\\
    df = pd.read_json(r'reviews_test.json')
    test = df.filter(['Score', 'Text'],axis=1)
    test['Score'] = test['Score'].apply(lambda r : "__label__" + str(r))
    test.to_csv(r'reviews_uncleaned_test.csv')

    # format: 	__label__<X>__label__<Y> ... <Text>
    with open('reviews_uncleaned_test.csv', mode='r', encoding='UTF-8') as f :
        test = f.read()
        lines = test.split('\n')
        lines.remove(lines[0])
        lines.remove(lines[-1]) 
     
        formatted_lines = []
        for line in lines :
            i = line.find(',')
            line = line[i+1:]
            
            # Assume that commas can appear in text field 
            j = line.find(',')
            line = list(line)
            line[j] = ' '
            line = ''.join(line)
            formatted_lines.append(line)

        test = '\n'.join(formatted_lines)
        final = open("reviews_uncleaned.test", mode='w+', encoding='UTF-8')
        final.write(test)
    


def main() :
    df = pd.read_json(r'reviews_train.json')
    print("reviews_train.json loaded:")
    print(df.head())

    # /////// PRE PROCESSING \\\\\\\\
    # Get the correct format for fasttext module
    format(df)
    format_uncleaned(df)

    # make and save test sets
    test()

    # Make and save the models
    print("m0: classifier trained on clean reviews")
    m0 = fasttext.train_supervised("reviews_cleaned.train", epoch=10, dim=50)
    #m1 = fasttext.train_supervised("reviews_cleaned.train", 'cbow', epoch=10, dim=50)
    print("m2: classifier trained on uncleaned reviews")
    m2 = fasttext.train_supervised("reviews_uncleaned.train", epoch=10, dim=50)
    #m3 = fasttext.train_supervised("reviews_uncleaned.train", 'cbow', epoch=10, dim=50)
    m4 = fasttext.train_supervised("reviews_cleaned.train", epoch=10, dim=50)
    m0.save_model('fasttext_skipgram_cleaned.bin')
    #m1.save_model('fasttext_cbow_cleaned.bin')
    m2.save_model('fasttext_skipgram_uncleaned.bin')
    #m3.save_model('fasttext_cbow_uncleaned.bin')



if __name__ == "__main__" :
    main()