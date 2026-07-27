# Word2vec model

Train a word2vec model (Mikolov et al., 2013) using a
[quanteda::tokens](https://quanteda.io/reference/tokens.html) object.

## Usage

``` r
textmodel_word2vec(
  x,
  dim = 50,
  type = c("cbow", "sg", "dm"),
  min_count = 5,
  window = ifelse(type == "sg", 10, 5),
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

  the architecture of the model; either "cbow" (continuous
  back-of-words), "sg" (skip-gram), or "dm" (distributed memory).

- min_count:

  the minimum frequency of the words. Words less frequent than this in
  `x` are removed before training.

- window:

  the size of the word window. Words within this window are considered
  to be the context of a target word.

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

Returns a textmodel_word2vec object with the following elements:

- values:

  a list of a matrix for word vector values.

- weights:

  a matrix for word vector weights.

- dim:

  the size of the word vectors.

- type:

  the architecture of the model.

- frequency:

  the frequency of words in `x`.

- window:

  the size of the word window.

- iter:

  the number of iterations in model training.

- alpha:

  the initial learning rate.

- use_ns:

  the use of negative sampling.

- ns_size:

  the size of negative samples.

- min_count:

  the value of min_count.

- concatenator:

  the concatenator in `x`.

- data:

  the original data supplied as `x` if `include_data = TRUE`.

- call:

  the command used to execute the function.

- version:

  the version of the wordvector package.

## Details

If `type = "dm"`, it trains a doc2vec model but saves only word vectors
to save storage space.
[textmodel_doc2vec](https://koheiw.github.io/wordvector/reference/textmodel_doc2vec.md)
should be used to access document vectors.

Users can changed the number of processors used for the parallel
computing via `options(wordvector_threads)`. When the value is large
than one, the result of every execution becomes slightly different even
if [`set.seed()`](https://rdrr.io/r/base/Random.html) is used because
parameters are updated in different orders by the processors.

## References

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013).
Distributed Representations of Words and Phrases and their
Compositionality. https://arxiv.org/abs/1310.4546.

## Examples

``` r
# \donttest{
library(quanteda)
library(wordvector)

# pre-processing
corp <- data_corpus_news2014 
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
   tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>% 
   tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                 padding = TRUE) %>% 
   tokens_tolower()

# train word2vec
wov <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, sample = 0.001)

# find similar words
head(similarity(wov, c("berlin", "germany", "france"), mode = "words"))
#>      berlin      germany        france    
#> [1,] "berlin"    "germany"      "france"  
#> [2,] "frankfurt" "braunschweig" "paris"   
#> [3,] "german"    "eintracht"    "germany" 
#> [4,] "germany"   "frankfurt"    "anglet"  
#> [5,] "amsterdam" "slovakia"     "bastia"  
#> [6,] "warsaw"    "hamburg"      "toulouse"
head(similarity(wov, c("berlin" = 1, "germany" = -1, "france" = 1), mode = "values"))
#>                   [,1]
#> somali     0.133980427
#> reporters -0.017841461
#> released   0.084553899
#> bail       0.111126832
#> still     -0.097537539
#> jailed     0.005625255
head(similarity(wov, analogy(~ berlin - germany + france), mode = "words"))
#>      [,1]      
#> [1,] "paris"   
#> [2,] "france"  
#> [3,] "french"  
#> [4,] "berlin"  
#> [5,] "brussels"
#> [6,] "hague"   
# }
```
