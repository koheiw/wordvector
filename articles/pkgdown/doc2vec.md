# doc2vec

In this article, **wordvector** is used to train
[doc2vec](https://doi.org/10.48550/arXiv.1405.4053) on **quanteda**’s
tokens objects.

## Prepare data

Download the corpus of news summaries.

``` r

# Load data
f <- tempfile()
download.file('https://www.dropbox.com/s/e19kslwhuu9yc2z/yahoo-news.RDS?dl=1', 
              f, mode = "wb")
```

``` r

library(quanteda)
## Package version: 4.5.1
## Unicode version: 17.0
## ICU version: 78.2
## Parallel computing: disabled
## See https://quanteda.io for tutorials and examples.
library(wordvector)
quanteda_options(verbose = TRUE)

# Construct corpus
dat <- readRDS(f)
dat$text <- paste0(dat$head, ". ", dat$body)
corp <- corpus(dat, text_field = 'text')

# Tokenize
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>% 
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                  padding = TRUE)
## Creating a tokens from a corpus object...
##  ...starting tokenization
##  ...tokenizing 1 of 7 blocks
##  ...preserving hyphens
##  ...preserving elisions
##  ...preserving social media tags (#, @)
##  ...tokenizing 2 of 7 blocks
##  ...preserving hyphens
##  ...preserving elisions
##  ...preserving social media tags (#, @)
##  ...tokenizing 3 of 7 blocks
##  ...preserving hyphens
##  ...preserving elisions
##  ...preserving social media tags (#, @)
##  ...tokenizing 4 of 7 blocks
##  ...preserving hyphens
##  ...preserving elisions
##  ...preserving social media tags (#, @)
##  ...tokenizing 5 of 7 blocks
##  ...preserving hyphens
##  ...preserving elisions
##  ...preserving social media tags (#, @)
##  ...tokenizing 6 of 7 blocks
##  ...preserving hyphens
##  ...preserving elisions
##  ...preserving social media tags (#, @)
##  ...tokenizing 7 of 7 blocks
##  ...preserving hyphens
##  ...preserving elisions
##  ...preserving social media tags (#, @)
##  ...removing separators, punctuation, symbols
##  ...298,565 unique types
##  ...complete, elapsed time: 95.1 seconds.
## Finished constructing tokens of 656,334 documents.
## Returning tokens of 656,334 documents (45,194,192 tokens, 298,002 types) from tokens_remove()
## Returning tokens of 656,334 documents (45,194,192 tokens, 239,782 types) from tokens_keep()
```

## Train doc2vec

[`textmodel_doc2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_doc2vec.md)
supports both `dm` (distributed memory) and `dbow` (distributed
bag-of-words) models.

``` r

# Set the number of processors
options(wordvector_threads = 16)

# Train doc2vec
dov <- textmodel_doc2vec(toks, dim = 50, type = "dm", min_count = 5, verbose = TRUE)
## Training distributed memory model with 50 dimensions
##  ...using 16 threads for distributed computing
##  ...initializing
##  ...negative sampling in 10 iterations
##  ......iteration 1 elapsed time: 14.60 seconds (alpha: 0.0459)
##  ......iteration 2 elapsed time: 30.32 seconds (alpha: 0.0413)
##  ......iteration 3 elapsed time: 45.92 seconds (alpha: 0.0369)
##  ......iteration 4 elapsed time: 63.19 seconds (alpha: 0.0322)
##  ......iteration 5 elapsed time: 79.57 seconds (alpha: 0.0276)
##  ......iteration 6 elapsed time: 95.23 seconds (alpha: 0.0232)
##  ......iteration 7 elapsed time: 114.04 seconds (alpha: 0.0180)
##  ......iteration 8 elapsed time: 130.24 seconds (alpha: 0.0134)
##  ......iteration 9 elapsed time: 144.77 seconds (alpha: 0.0093)
##  ......iteration 10 elapsed time: 159.28 seconds (alpha: 0.0052)
##  ...complete
```

Since the distributed memory model has hidden layers for documents and
words, you can extract document and word vectors using
[`as.matrix()`](https://rdrr.io/r/base/matrix.html)

``` r

# Extract document vector
dim(as.matrix(dov, layer = "documents"))
## [1] 656334     50

# Extract word vector
dim(as.matrix(dov, layer = "words"))
## [1] 79259    50
```

## Predict probability

If `probabitliy()` is applied to a fitted doc2vec model, you receive the
predicted probability of the words in each document.

``` r

head(probability(dov, c("bad", "good"), mode = "numeric", layer = "documents"))
##             bad      good
## text1 0.2031377 0.3671307
## text2 0.2567344 0.2236566
## text3 0.1465683 0.0752158
## text4 0.5442547 0.8330752
## text5 0.4162880 0.3955734
## text6 0.4259072 0.4884102
```
