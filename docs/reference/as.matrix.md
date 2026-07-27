# Extract word or document vectors

Extract word or document vectors from a `textmodel_word2vec` or
`textmodel_doc2vec` object.

## Usage

``` r
# S3 method for class 'textmodel_doc2vec'
as.matrix(
  x,
  normalize = TRUE,
  layer = c("documents", "words"),
  group = FALSE,
  ...
)

# S3 method for class 'textmodel_word2vec'
as.matrix(x, normalize = TRUE, layer = "words", padding = FALSE, ...)
```

## Arguments

- x:

  a `textmodel_word2vec` or `textmodel_doc2vec` object.

- normalize:

  if `TRUE`, returns normalized vectors.

- layer:

  the layer from which the vectors are extracted.

- group:

  \[experimental\] average sentence or paragraph vectors from the same
  document. Silently ignored when `layer = "words"`.

- ...:

  not used.

- padding:

  if `TRUE`, add a row with zeros before the word vectors.

## Value

a matrix that contain the word or document vectors in rows.
