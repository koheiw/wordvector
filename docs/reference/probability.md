# Compute probability of words

Compute the probability of words given other words.

## Usage

``` r
probability(
  x,
  targets,
  layer = c("words", "documents"),
  mode = c("character", "numeric"),
  ...
)
```

## Arguments

- x:

  a trained `textmodel_wordvector` object.

- targets:

  words for which probabilities are computed.

- layer:

  the layer based on which probabilities are computed.

- mode:

  specify the type of resulting object.

- ...:

  passed to [`as.matrix()`](https://rdrr.io/r/base/matrix.html).

## Value

a matrix of words or documents sorted in descending order by the
probability scores when `mode = "character"`; a matrix of the
probability scores when `mode = "numeric"`. When `targets` is a named
numeric vector, probability scores are weighted by the values.

## See also

[`similarity()`](https://koheiw.github.io/wordvector/reference/similarity.md)
