# Compute similarity between word or document vectors

Compute the cosine similarity between word vectors for selected words.

## Usage

``` r
similarity(
  x,
  targets,
  layer = c("words", "documents"),
  mode = c("character", "numeric")
)
```

## Arguments

- x:

  a `textmodel_wordvector` object.

- targets:

  words or documents for which similarity is computed.

- layer:

  the layer based on which similarity is computed. This must be
  "documents" when `targets` are document names.

- mode:

  specify the type of resulting object.

## Value

a `matrix` of cosine similarity scores when `mode = "numeric"` or of
words sorted in descending order by the similarity scores when
`mode = "character"`. When `targets` is a named numeric vector, word (or
document) vectors are weighted and summed before computing similarity
scores.

## See also

[`probability()`](https://koheiw.github.io/wordvector/reference/probability.md)
