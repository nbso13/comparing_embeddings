import fasttext as ft
import export_wordvecs_fasttext as ev

mcs,mus = ([mc0,mc1,mc2,mc3,mc4],[mu0,mu1,mu2,mu3,mu4]) = ft.main()

# the model we care about it mc1

labels, reviews, vectors = ev.get_vectors(mc1)