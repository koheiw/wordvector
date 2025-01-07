## Changes in v0.2.0

- Rename `word2vec()`, `doc2vec()` and `lsa()` to `textmodel_word2vec()`, `textmodel_doc2vec()` and `textmodel_lsa()` respectively. 
- Simplify the C++ code to make maintenance easier.
- Add `normalize` to `word2vec` to disable or enable word vector normalization.
- Add `weights()` to extract back-propagation weights.
- Make `analogy()` to convert a formula to named character vector.
- Improve the stability of `word2vec()` when `verbose = TRUE`.

## Changes in v0.1.0

- Fork https://github.com/bnosac/word2vec and change the package name to wordvector.
- Replace a list of character with **quanteda**'s tokens object as an input object.
- Recreate `word2vec()` with new argument names and object structures.
- Create `lda()` to train word vectors using Latent Semantic Analysis.
- Add `similarity()` and `analogy()` functions using **proxyC**.
- Add `data_corpus_news2014` that contain 20,000 news summaries as package data.