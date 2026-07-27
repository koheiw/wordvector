# Doc2vec model

Train a doc2vec model (Le & Mikolov, 2014) using a
[quanteda::tokens](https://quanteda.io/reference/tokens.html) object.

## Usage

``` r
textmodel_doc2vec(
  x,
  dim = 50,
  type = c("dm", "dbow"),
  min_count = 5,
  window = 5,
  iter = 10,
  alpha = 0.05,
  model = NULL,
  use_ns = TRUE,
  ns_size = 5,
  sample = 0.001,
  tolower = TRUE,
  include_data = FALSE,
  verbose = FALSE,
  ...
)
```

## Arguments

- x:

  a [quanteda::tokens](https://quanteda.io/reference/tokens.html) or
  [quanteda::tokens_xptr](https://quanteda.io/reference/tokens_xptr.html)
  object.

- dim:

  the size of the word vectors.

- type:

  the architecture of the model; either "dm" (distributed memory) or
  "dbow" (distributed bag-of-words).

- min_count:

  the minimum frequency of the words. Words less frequent than this in
  `x` are removed before training.

- window:

  the size of the window for context words. Ignored when `type = "dbow"`
  as its context window is the entire document (sentence or paragraph).

- iter:

  the number of iterations in model training.

- alpha:

  the initial learning rate.

- model:

  a trained Word2vec model; if provided, its word vectors are updated
  for `x`.

- use_ns:

  if `TRUE`, negative sampling is used. Otherwise, hierarchical softmax
  is used.

- ns_size:

  the size of negative samples. Only used when `use_ns = TRUE`.

- sample:

  the rate of sampling of words based on their frequency. Sampling is
  disabled when `sample = 1.0`

- tolower:

  lower-case all the tokens before fitting the model.

- include_data:

  if `TRUE`, the resulting object includes the data supplied as `x`.

- verbose:

  if `TRUE`, print the progress of training.

- ...:

  additional arguments.

## Value

Returns a textmodel_doc2vec object with matrices for word and document
vector values,
[quanteda::docvars](https://quanteda.io/reference/docvars.html) and
[quanteda::ntoken](https://quanteda.io/reference/ntoken.html) of `x`.
Other elements are the same as
[textmodel_word2vec](https://koheiw.github.io/wordvector/reference/textmodel_word2vec.md).

## References

Le, Q. V., & Mikolov, T. (2014). Distributed Representations of
Sentences and Documents (No. arXiv:1405.4053). arXiv.
https://doi.org/10.48550/arXiv.1405.4053
