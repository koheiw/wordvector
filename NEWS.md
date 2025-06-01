## Change in v0.5.1

- Normalize document vectors by the lengths of documents when `normalize = TRUE`.
- Add `weights` to `textmodel_doc2vec()` to adjust the salience of words in the document vectors.
- Add `include_data` to `textmodel_word2vec()` to save the original tokens object.

## Changes in v0.5.0

- Add the `model` argument to `textmodel_word2vec()` to update existing models.
- The `normalize` argument is moved from `textmodel_word2vec()` to `as.matrix()`. The original argument is deprecated and set to `FALSE` by default. 
- Remove `weights()`.
- Improve the structure of C++ code.

## Changes in v0.4.0

- Add the `tolower` argument and set to `TRUE` to lower-case tokens.
- Allow `x` to be quanteda's tokens_xptr object to enhance efficiency.

## Changes in v0.3.0

- Save docvars in the `textmodel_doc2vec` objects.
- Set zero for empty documents in the `textmodel_doc2vec` objects. 
- Add `probability()` to compute probability of words.

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