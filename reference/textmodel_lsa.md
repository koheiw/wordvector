# Latent Semantic Analysis model

Train a Latent Semantic Analysis model (Deerwester et al., 1990) on a
[quanteda::tokens](https://quanteda.io/reference/tokens.html) object.

## Usage

``` r
textmodel_lsa(
  x,
  dim = 50,
  min_count = 5L,
  engine = c("RSpectra", "irlba", "rsvd"),
  weight = "count",
  tolower = TRUE,
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

- min_count:

  the minimum frequency of the words. Words less frequent than this in
  `x` are removed before training.

- engine:

  select the engine perform SVD to generate word vectors.

- weight:

  weighting scheme passed to
  [`quanteda::dfm_weight()`](https://quanteda.io/reference/dfm_weight.html).

- tolower:

  if `TRUE` lower-case all the tokens before fitting the model.

- verbose:

  if `TRUE`, print the progress of training.

- ...:

  additional arguments.

## Value

Returns a textmodel_wordvector object with the following elements:

- values:

  a matrix for word vectors values.

- weights:

  a matrix for word vectors weights.

- frequency:

  the frequency of words in `x`.

- engine:

  the SVD engine used.

- weight:

  weighting scheme.

- min_count:

  the value of min_count.

- concatenator:

  the concatenator in `x`.

- call:

  the command used to execute the function.

- version:

  the version of the wordvector package.

## References

Deerwester, S. C., Dumais, S. T., Landauer, T. K., Furnas, G. W., &
Harshman, R. A. (1990). Indexing by latent semantic analysis. JASIS,
41(6), 391–407.

## Examples

``` r
# \donttest{
library(quanteda)
#> Package version: 4.4
#> Unicode version: 15.1
#> ICU version: 74.2
#> Parallel computing: disabled
#> See https://quanteda.io for tutorials and examples.
library(wordvector)

# pre-processing
corp <- corpus_reshape(data_corpus_news2014)
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
   tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>% 
   tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                 padding = TRUE) %>% 
   tokens_tolower()

# train LSA
lsa <- textmodel_lsa(toks, dim = 50, min_count = 5, verbose = TRUE)
#> Performing SVD into 50 dimensions
#> ...using RSpectra
#> ...complete

# find similar words
head(similarity(lsa, c("berlin", "germany", "france"), mode = "words"))
#>      berlin       germany     france       
#> [1,] "berlin"     "germany"   "france"     
#> [2,] "german"     "france"    "montpellier"
#> [3,] "warsaw"     "closer"    "paris"      
#> [4,] "timmermans" "berlin"    "froome"     
#> [5,] "germany"    "moscovici" "germany"    
#> [6,] "lisbon"     "tougher"   "french"     
head(similarity(lsa, c("berlin" = 1, "germany" = -1, "france" = 1), mode = "values"))
#>                  [,1]
#> somali    -0.05066583
#> reporters -0.01372003
#> released   0.01725500
#> bail       0.08066897
#> still      0.09478653
#> jailed     0.02431590
head(similarity(lsa, analogy(~ berlin - germany + france)))
#>      [,1]       
#> [1,] "paris"    
#> [2,] "berlin"   
#> [3,] "koscielny"
#> [4,] "mans"     
#> [5,] "nibali"   
#> [6,] "hertha"   
# }
```
