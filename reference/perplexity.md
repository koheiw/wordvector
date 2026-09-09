# \[experimental\] Compute perplexity of a model

Compute the perplexity of a trained word2vec model with data.

## Usage

``` r
perplexity(x, targets, data, layer = c("words", "documents"))
```

## Arguments

- x:

  a trained `textmodel_wordvector` object.

- data:

  a [quanteda::tokens](https://quanteda.io/reference/tokens.html) or
  [quanteda::dfm](https://quanteda.io/reference/dfm.html); the
  probabilities of words are tested against occurrences of words in it.
