## Changes in v0.1

- Fork https://github.com/bnosac/word2vec and change the package name to wordvector.
- Replace a list of character with **quanteda**'s tokens object as an input object.
- Recreate `word2vec()` with new argument names and object structures.
- Create `lda()` to train word vectors using Latent Semantic Analysis.
- Add `similarity()` and `analogy()` functions using **proxyC**.