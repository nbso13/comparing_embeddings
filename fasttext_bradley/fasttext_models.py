import pandas as pd
import numpy as np
import scipy
import fasttext

'''
Run this file to make text files fromatted for training FastText models,
the trained models themselves (saved in /models), and run a series of
tests that compare the accuracy of the different models.
'''

BAD_WORDS = ['<','>','br','</s>','<s>','/><br']
EPOCHS = 10
DIMENSIONS = 15
MINCOUNT = 6000

def format_uncleaned(df) :
    '''
    Formats the training set of uncleaned data
    Saves as reviews_uncleaned.train
    '''
    train = df.filter(['Score','Text'], axis=1)
    train['Score'] =train['Score'].apply(lambda r : "__label__" + str(r))

    train.to_csv(r'data/reviews_unclean_train.csv')

    # format: 	__label__<X>__label__<Y> ... <Text>
    # encoding must be UTF-8
    with open('data/reviews_unclean_train.csv', mode='r', encoding='UTF-8') as f :
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

            # Get rid of HTML markdown tokens
            splitline = line.split()
            for word in BAD_WORDS :
                while word in splitline :
                    splitline.remove(word)
            line = ' '.join(splitline)

        train = '\n'.join(formatted_lines)

        print("Preprocessed uncleaned data:")
        final = open("data/reviews_uncleaned.train", mode='w+', encoding='UTF-8')
        final.write(train)




def format(df) :
    '''
    Formats the training set of cleaned data
    Saves as reviews_cleaned.train
    '''
    print()
    print('Preprocessing the data:')

    train = df.filter(['Score', 'clean'],axis=1)

    # Star ratings are labels with format "__<label>__#Score"
    train['Score'] =train['Score'].apply(lambda r : "__label__" + str(r))

    # Output as text file with proper format
    # First output as csv, then read back in as txt because
    # the csv's plaintext is easier to manipulate
    train.to_csv(r'data/reviews_train.csv')

    # format: 	__label__<X>__label__<Y> ... <Text>
    # encoding must be UTF-8
    with open('data/reviews_train.csv', mode='r', encoding='UTF-8') as f :
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

            # Get rid of HTML markdown tokens
            splitline = line.split()
            for word in BAD_WORDS :
                while word in splitline :
                    splitline.remove(word)
            line = ' '.join(splitline)

        train = '\n'.join(formatted_lines)

        print("Preprocessed cleaned data:")
        print(train[:500])

        final = open("data/reviews_cleaned.train", mode='w+', encoding='UTF-8')
        final.write(train)



def make_tests() :
    '''
    Formats the testing set of both cleaned and uncleaned data
    Saves one as reviews_cleaned.test
    Saves other as reviews_uncleaned.test
    '''
    # ///////// Build cleaned test set \\\\\\\\\\
    df = pd.read_json(r'data/reviews_test.json')
    test = df.filter(['Score', 'clean'],axis=1)
    test['Score'] = test['Score'].apply(lambda r : "__label__" + str(r))
    test.to_csv(r'data/reviews_test.csv')

    # format: 	__label__<X>__label__<Y> ... <Text>
    with open('data/reviews_test.csv', mode='r', encoding='UTF-8') as f :
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

            # Not sure if I should remove HTML markdown tokens, so I will not
            # to keep the integrity of the test data.

        test = '\n'.join(formatted_lines)
        final = open("data/reviews_cleaned.test", mode='w+', encoding='UTF-8')
        final.write(test)
    
    # //////// Build uncleaned test set \\\\\\\\\
    df = pd.read_json(r'data/reviews_test.json')
    test = df.filter(['Score', 'Text'],axis=1)
    test['Score'] = test['Score'].apply(lambda r : "__label__" + str(r))
    test.to_csv(r'data/reviews_uncleaned_test.csv')

    # format: 	__label__<X>__label__<Y> ... <Text>
    with open('data/reviews_uncleaned_test.csv', mode='r', encoding='UTF-8') as f :
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
        final = open("data/reviews_uncleaned.test", mode='w+', encoding='UTF-8')
        final.write(test)
    


def main(arg) :
    df = pd.read_json(r'data/reviews_train.json')
    print("reviews_train.json loaded:")
    print(df.head())

    # /////// PRE PROCESSING \\\\\\\\
    # Get the correct format for fasttext module's training dataset
    print("format()")
    format(df)
    print("unclean format()")

    format_uncleaned(df)

    print("Making tests")
    # make and save test sets
    make_tests()
    print("Tests made.")

    if arg == 'epochs' :
        # Make and save the models
        # Varying word vector dimension by 25, 50, 100, 200, 300
        # Varying training data (cleaned and uncleaned)
        mc0 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=1, dim=DIMENSIONS, minCount=MINCOUNT)
        mc1 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=2, dim=DIMENSIONS, minCount=MINCOUNT)
        mc2 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=3, dim=DIMENSIONS, minCount=MINCOUNT)
        mc3 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=4, dim=DIMENSIONS, minCount=MINCOUNT)
        mc4 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=5, dim=DIMENSIONS, minCount=MINCOUNT)
        mc5 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=6, dim=DIMENSIONS, minCount=MINCOUNT)
        mc6 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=7, dim=DIMENSIONS, minCount=MINCOUNT)
        mc7 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=8, dim=DIMENSIONS, minCount=MINCOUNT)
        mc8 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=9, dim=DIMENSIONS, minCount=MINCOUNT)
        mc9 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=10, dim=DIMENSIONS, minCount=MINCOUNT)
        mc10 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=11, dim=DIMENSIONS, minCount=MINCOUNT)
        mc11 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=12, dim=DIMENSIONS, minCount=MINCOUNT)
        mc12 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=13, dim=DIMENSIONS, minCount=MINCOUNT)
        mc13 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=14, dim=DIMENSIONS, minCount=MINCOUNT)
        mc14 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=15, dim=DIMENSIONS, minCount=MINCOUNT)
        ms = [mc0,mc1,mc2,mc3,mc4,mc5,mc6,mc7,mc8,mc9,mc10,mc11,mc12,mc13,mc14]
        return ms
    else :
        # Make and save the models
        # Varying word vector dimension by 25, 50, 100, 200, 300
        # Varying training data (cleaned and uncleaned)
        print("mc0: classifier trained on clean reviews, 10 epochs, vector-size 25.")    
        mc0 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=EPOCHS, dim=25, minCount=MINCOUNT)

        print("mc1: classifier trained on clean reviews, 10 epochs, vector-size 50.")    
        mc1 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=EPOCHS, dim=50, minCount=MINCOUNT)

        print("mc2: classifier trained on clean reviews, 10 epochs, vector-size 100.")    
        mc2 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=EPOCHS, dim=100, minCount=MINCOUNT)

        print("mc3: classifier trained on clean reviews, 10 epochs, vector-size 200.")    
        mc3 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=EPOCHS, dim=200, minCount=MINCOUNT)

        print("mc4: classifier trained on clean reviews, 10 epochs, vector-size 300.")    
        mc4 = fasttext.train_supervised("data/reviews_cleaned.train", epoch=EPOCHS, dim=300, minCount=MINCOUNT)

        mcs = [mc0,mc1,mc2,mc3,mc4]

        print("mu0: classifier trained on unclean reviews, 10 epochs, vector-size 25.")    
        mu0 = fasttext.train_supervised("data/reviews_uncleaned.train", epoch=EPOCHS, dim=25, minCount=MINCOUNT)
        
        print("mu1: classifier trained on unclean reviews, 10 epochs, vector-size 50.")    
        mu1 = fasttext.train_supervised("data/reviews_uncleaned.train", epoch=EPOCHS, dim=50, minCount=MINCOUNT)
        
        print("mu2: classifier trained on unclean reviews, 10 epochs, vector-size 100.")    
        mu2 = fasttext.train_supervised("data/reviews_uncleaned.train", epoch=EPOCHS, dim=100, minCount=MINCOUNT)
        
        print("mu3: classifier trained on unclean reviews, 10 epochs, vector-size 200.")    
        mu3 = fasttext.train_supervised("data/reviews_uncleaned.train", epoch=EPOCHS, dim=200, minCount=MINCOUNT)
        
        print("mu4: classifier trained on unclean reviews, 10 epochs, vector-size 300.")    
        mu4 = fasttext.train_supervised("data/reviews_uncleaned.train", epoch=EPOCHS, dim=300, minCount=MINCOUNT)
        print('Models made.')

        mus = [mu0,mu1,mu2,mu3,mu4]
    
        # Save all 10 models
        Ds = ['25','50','100','200','300']
        d = 0
        print(d)
        for m in mcs :
            m.save_model('data/fasttext_skipgram_cleaned_D' + Ds[d] +'.bin')
            d+=1
            print(m.get_dimension())
        
        d = 0
        for m in mus :
            m.save_model('data/fasttext_skipgram_uncleaned_D' + Ds[d] +'.bin')
            d+=1
        print('Models saved.')

        return (mcs, mus)


if __name__ == "__main__" :
    main()