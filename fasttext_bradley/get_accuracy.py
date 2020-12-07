import fasttext
import pandas as pd 

'''
To get the accuracy for a fasttext calssifier on the test set, call get_accuracy(m), 
where m is the string name of the corresponding model described in fasttext_evaluation.ipynb

Unfortunately, this is the easiest way to do this. 
Take FastText's word for it: https://github.com/facebookresearch/fastText/issues/261

def run_tests(m, labels, reviews) :
    if len(labels) != len(reviews) :
        print("Error: get_accuracy, run_test : labels different length than reviews")

    hits = 0
    n = len(labels)

    for i in range(len(labels)) :
        guess = m.predict(reviews[i])
        # retrieve guess, format : (('__label__5',), array([0.72070283]))
        guess = guess[0][0]
        if guess == labels[i] :
            hits += 1

    return hits / n
        

def get_accuracy(m) :
    # load data

    test = open("reviews_uncleaned.test", mode='r', encoding='UTF-8')
    test = test.read()

    labels = []
    reviews = []
    test = test.split('\n')
    for review in test :
        # split : ["__label__","Review text ..."]
        temp = review.split(' ',1)
        labels.append(temp[0])
        reviews.append(temp[1])

    # run tests and get output
    return run_tests(m, labels, reviews)
'''

def load_model(m) :
        if (m == "mc0") :
            ret = fasttext.load_model('data/fasttext_skipgram_cleaned_D25.bin')
        elif(m == "mc1") :
            ret = fasttext.load_model('data/fasttext_skipgram_cleaned_D50.bin')
        elif(m == "mc2") :
            ret = fasttext.load_model('data/fasttext_skipgram_cleaned_D100.bin')
        elif(m == "mc3") :
            ret = fasttext.load_model('data/fasttext_skipgram_cleaned_D200.bin')
        elif(m == "mc4") :
            ret = fasttext.load_model('data/fasttext_skipgram_cleaned_D300.bin')
        elif(m == "mu0") :
            ret = fasttext.load_model('data/fasttext_skipgram_uncleaned_D25.bin')
        elif(m == "mu1") :
            ret = fasttext.load_model('data/fasttext_skipgram_uncleaned_D50.bin')
        elif(m == "mu2") :
            ret = fasttext.load_model('data/fasttext_skipgram_uncleaned_D100.bin')
        elif(m == "mu3") :
            ret = fasttext.load_model('data/fasttext_skipgram_uncleaned_D200.bin')
        elif(m == "mu4") :
            ret = fasttext.load_model('data/fasttext_skipgram_uncleaned_D300.bin')
        else :
            print("ERROR: load_model() in make_vectors() : Input must be format 'mc#' or 'mu#'")
        return ret



def run_tests(m, labels, reviews) :
    if len(labels) != len(reviews) :
        print("Error: get_accuracy, run_test : labels different length than reviews")

    hits = 0
    n = len(labels)

    for i in range(len(labels)) :
        guess = m.predict(reviews[i])
        # retrieve guess, format : (('__label__5',), array([0.72070283]))
        guess = guess[0][0]
        if guess == reviews[i] :
            hits += 1

    return hits / n
    


def get_accuracy(m) :
    # load data
    m = load_model(m)

    test = open("data/reviews_uncleaned.test", mode='r', encoding='UTF-8')
    test = test.read()

    labels = []
    reviews = []
    test = test.split('\n')
    for review in test :
        # split : ["__label__","Review text ..."]
        temp = review.split(' ',1)
        labels.append(temp[0])
        reviews.append(temp[1])

    # run tests and get output
    return run_tests(m, labels, review)