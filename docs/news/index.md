# Changelog

## Changes in v0.6.3

- Add `padding` to [`as.matrix()`](https://rdrr.io/r/base/matrix.html)
  for using word2vec as a pre-trained model in **torch**.
- Update internal functions for **quanteda** v4.5.0.

## Changes in v0.6.2

CRAN release: 2026-04-06

- Add `layer` to
  [`perplexity()`](https://koheiw.github.io/wordvector/reference/perplexity.md)
  for `textmodel_doc2vec` models.
- Save document lengths as `ntoken` in trained `textmodel_doc2vec`
  models.
- Update `as.textmode_doc2vec()` to save output layer weights.
- Update tests for **quanteda** v4.4.0.

## Changes in v0.6.1

CRAN release: 2026-02-25

- Mention doc2vec in package description.
- Add
  [`perplexity()`](https://koheiw.github.io/wordvector/reference/perplexity.md)
  to asses models’ the goodness-of-fit to data.
- Save **quanteda**’s internal docvars in the `textmodel_doc2vec`
  objects.
- Add `group` to [`as.matrix()`](https://rdrr.io/r/base/matrix.html) to
  average sentence or paragraph vectors from the same documents.

## Changes in v0.6.0

CRAN release: 2025-12-09

- Upgrade `textmodel_doc2vec` to train the distributed memory (DM) and
  distributed bag-of-word (DBOW) models.
- Add
  [`as.textmodel_doc2vec()`](https://koheiw.github.io/wordvector/reference/as.textmodel_doc2vec.md)
  to create document vectors as weighted average of word vectors.
- Add `layer` to [`as.matrix()`](https://rdrr.io/r/base/matrix.html) to
  choose between word or document vectors.
- `normalize` is now defunct in
  [`textmodel_word2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_word2vec.md).

## Changes in v0.5.1

CRAN release: 2025-06-20

- Add `normalize` to
  [`textmodel_doc2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_doc2vec.md)
  and pass it to [`as.matrix()`](https://rdrr.io/r/base/matrix.html).
- Add `weights` to
  [`textmodel_doc2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_doc2vec.md)
  to adjust the salience of words in the document vectors.
- Add `include_data` to
  [`textmodel_word2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_word2vec.md)
  to save the original tokens object.

## Changes in v0.5.0

CRAN release: 2025-05-15

- Add the `model` argument to
  [`textmodel_word2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_word2vec.md)
  to update existing models.
- The `normalize` argument is moved from
  [`textmodel_word2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_word2vec.md)
  to [`as.matrix()`](https://rdrr.io/r/base/matrix.html). The original
  argument is deprecated and set to `FALSE` by default.
- Remove [`weights()`](https://rdrr.io/r/stats/weights.html).
- Improve the structure of C++ code.

## Changes in v0.4.0

- Add the `tolower` argument and set to `TRUE` to lower-case tokens.
- Allow `x` to be quanteda’s tokens_xptr object to enhance efficiency.

## Changes in v0.3.0

CRAN release: 2025-03-12

- Save docvars in the `textmodel_doc2vec` objects.
- Set zero for empty documents in the `textmodel_doc2vec` objects.
- Add
  [`probability()`](https://koheiw.github.io/wordvector/reference/probability.md)
  to compute probability of words.

## Changes in v0.2.0

CRAN release: 2025-01-07

- Rename `word2vec()`, `doc2vec()` and `lsa()` to
  [`textmodel_word2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_word2vec.md),
  [`textmodel_doc2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_doc2vec.md)
  and
  [`textmodel_lsa()`](https://koheiw.github.io/wordvector/reference/textmodel_lsa.md)
  respectively.
- Simplify the C++ code to make maintenance easier.
- Add `normalize` to `word2vec` to disable or enable word vector
  normalization.
- Add [`weights()`](https://rdrr.io/r/stats/weights.html) to extract
  back-propagation weights.
- Make
  [`analogy()`](https://koheiw.github.io/wordvector/reference/analogy.md)
  to convert a formula to named character vector.
- Improve the stability of `word2vec()` when `verbose = TRUE`.

## Changes in v0.1.0

CRAN release: 2024-12-11

- Fork <https://github.com/bnosac/word2vec> and change the package name
  to wordvector.
- Replace a list of character with **quanteda**’s tokens object as an
  input object.
- Recreate `word2vec()` with new argument names and object structures.
- Create `lda()` to train word vectors using Latent Semantic Analysis.
- Add
  [`similarity()`](https://koheiw.github.io/wordvector/reference/similarity.md)
  and
  [`analogy()`](https://koheiw.github.io/wordvector/reference/analogy.md)
  functions using **proxyC**.
- Add `data_corpus_news2014` that contain 20,000 news summaries as
  package data.
