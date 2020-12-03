'''
This module contains a function that returns a 2-tuple = (fasttext classifier, the training labels, the review texts)
for use in sci kit's ML functions.

The input for make_vectors is the string name of the corresponding model described in fasttext_evaluation.ipynb.



I am basing it off of:
https://stackoverflow.com/questions/49236166/how-to-make-use-of-pre-trained-word-embeddings-when-training-a-model-in-sklearn
to be used with sklearn.
    pseudo-code that may work:   
        import export_wordvecs_fasttext as wv

        word_vecs_matrix = wv.make_vectors(model)    # model = e.g. mu1

        def transform(x):
            return word_vecs_matrix[x]
        transformer = FunctionTransformer(transform)

        # the transformer can be passed into a logistic regression with Pipeline
        # (I am fuzzy on the details of this)
        pip = Pipeline([(Countvectorizer()), (TfidfTransformer()), (Classifier())])
        pip.fit(X_train, y_train)

Before using these functions,  user should make the models in fasttext_evaluation.ipynb
'''

'''
It may be useful for you to glance at the model object, in case you need to edit the code and
get different information from the models.

from https://fasttext.cc/docs/en/python-module.html :
    get_dimension           # Get the dimension (size) of a lookup vector (hidden layer).
                            # This is equivalent to `dim` property.
    get_input_vector        # Given an index, get the corresponding vector of the Input Matrix.
*   get_input_matrix        # Get a copy of the full input matrix of a Model.
    get_labels              # Get the entire list of labels of the dictionary
                            # This is equivalent to `labels` property.
    get_line                # Split a line of text into words and labels.
    get_output_matrix       # Get a copy of the full output matrix of a Model.
*   get_sentence_vector     # Given a string, get a single vector represenation. This function
                            # assumes to be given a single line of text. We split words on
                            # whitespace (space, newline, tab, vertical tab) and the control
                            # characters carriage return, formfeed and the null character.
    get_subword_id          # Given a subword, return the index (within input matrix) it hashes to.
    get_subwords            # Given a word, get the subwords and their indicies.
    get_word_id             # Given a word, get the word id within the dictionary.
    get_word_vector         # Get the vector representation of word.
    get_words               # Get the entire list of words of the dictionary
                            # This is equivalent to `words` property.
    is_quantized            # whether the model has been quantized
    predict                 # Given a string, get a list of labels and a list of corresponding probabilities.
    quantize                # Quantize the model reducing the size of the model and it's memory footprint.
    save_model              # Save the model to the given path
    test                    # Evaluate supervised model using file given by path
    test_label              # Return the precision and recall score for each label.    
'''
import fasttext
import numpy as np 
import pandas as pd 


'''
str_m = string name of the corresponding model described in fasttext_evaluation.ipynb
'''
def make_vectors(str_m) :
    def remove_labels(set) :
        # ( Recall the format for fasttext inputs: __label__<X>__label__<Y> ... <Text> )
        # split the label from the text in the dataset and return tuple (labels, reviews)
        # (labels, reviews) == (list of labels, list of review text)
        labels = []
        reviews = []

        # split on each label+review pair
        set.split('\n')
        for line in set :
            # split label from review
            line.split(' ', 1)
            labels.append(line[0])
            reviews.append(line[1])
        return (labels, reviews)

    def return_vectors(m, labels, reviews) :
        # returns a 3-tuple (labels, reviews, embeddings)
        # (labels, reviews, embeddings) == (list of labels, list of review texts, list of review text vectors)
        # return_vectors really just generates vectors for each review and populates a list that matches on index.
        embeddings = []
        for review in reviews :
            embeddings.append(m.get_sentence_vector(review))

        return (labels, reviews, embeddings)

    def load_model(m) :
        if (m == "mc0") :
            ret = fasttext.load_model('fasttext_skipgram_cleaned_D25.bin')
        elif(m == "mc1") :
            ret = fasttext.load_model('fasttext_skipgram_cleaned_D50.bin')
        elif(m == "mc2") :
            ret = fasttext.load_model('fasttext_skipgram_cleaned_D100.bin')
        elif(m == "mc3") :
            ret = fasttext.load_model('fasttext_skipgram_cleaned_D200.bin')
        elif(m == "mc4") :
            ret = fasttext.load_model('fasttext_skipgram_cleaned_D300.bin')
        elif(m == "mu0") :
            ret = fasttext.load_model('fasttext_skipgram_uncleaned_D25.bin')
        elif(m == "mu1") :
            ret = fasttext.load_model('fasttext_skipgram_uncleaned_D50.bin')
        elif(m == "mu2") :
            ret = fasttext.load_model('fasttext_skipgram_uncleaned_D100.bin')
        elif(m == "mu3") :
            ret = fasttext.load_model('fasttext_skipgram_uncleaned_D200.bin')
        elif(m == "mu4") :
            ret = fasttext.load_model('fasttext_skipgram_uncleaned_D300.bin')
        else :
            print("ERROR: load_model() in make_vectors() : Input must be format 'mc#' or 'mu#'")
        return ret

    m = load_model(str_m)

    if (str_m[1] == 'c') :
        train = open("reviews_cleaned.train", mode='r', encoding='UTF-8')
        labels, reviews = remove_labels(train)
    elif (str_m[1] == 'u') :
        train = open("reviews_uncleaned.train", mode='r', encoding='UTF-8')
        labels, reviews = remove_labels(train)
    
    (labels, reviews, vectors) = return_vectors(m, labels, reviews)

    return (labels, reviews, vectors)