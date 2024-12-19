## Changes in v0.1.1

- Update `analogy()` to use vectors instead of formulas.
- Improve the stability of `word2vec()` when `verbose = TRUE`.

## Changes in v0.1.0

- Fork https://github.com/bnosac/word2vec and change the package name to wordvector.
- Replace a list of character with **quanteda**'s tokens object as an input object.
- Recreate `word2vec()` with new argument names and object structures.
- Create `lda()` to train word vectors using Latent Semantic Analysis.
- Add `similarity()` and `analogy()` functions using **proxyC**.
- Add `data_corpus_news2014` that contain 20,000 news summaries as package data.