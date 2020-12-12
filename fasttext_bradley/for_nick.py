import fasttext as ft
import export_wordvecs_fasttext as ev
import numpy as np
import get_accuracy as ga

mc1 = ft.load_model('data/fasttext_skipgram_cleaned_D50.bin')


# the model we care about it mc1
labels, reviews, vectors = ev.get_vectors(mc1)

a,lables,reviews = ga.get_accuracy(mc1)

test_vecs = []
for r in reviews :
    test_vecs.append(mc1.get_sentence_vector(r))



test_vecs = np.stack(test_vecs, axis=0)
vectors = np.stack(vectors, axis=0)

np.save('fasttext_vectors',vectors)
np.save('fasttext_test_vectors',test_vecs)
