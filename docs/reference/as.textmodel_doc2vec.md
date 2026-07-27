# Create distributed representation of documents

Create distributed representation of documents as weighted word vectors.

## Usage

``` r
as.textmodel_doc2vec(x, model, normalize = FALSE, group_data = FALSE, ...)
```

## Arguments

- x:

  a [quanteda::tokens](https://quanteda.io/reference/tokens.html) or
  [quanteda::dfm](https://quanteda.io/reference/dfm.html) object.

- model:

  a textmodel_wordvector object.

- normalize:

  if `TRUE`, normalized word vectors before creating document vectors.

- group_data:

  if `TRUE`, apply `dfm_group(x)` before creating document vectors.

- ...:

  additional arguments passed to
  [quanteda::object2id](https://quanteda.io/reference/object2id.html).

## Value

Returns a textmodel_docvector object with the following elements:

- values:

  a list of matrices for word and document vectors.

- dim:

  the size of the document vectors.

- concatenator:

  the concatenator in `x`.

- docvars:

  document variables copied from `x`.

- normalize:

  if the document vectors are normalized.

- call:

  the command used to execute the function.

- version:

  the version of the wordvector package.
