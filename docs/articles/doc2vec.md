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
##  ...complete, elapsed time: 90.4 seconds.
## Finished constructing tokens from 656,334 documents
## tokens_remove() changed from 298,565 types (656,334 documents, 45,194,192 tokens) to 298,002 types (656,334 documents, 28,232,774 tokens)
## tokens_keep() changed from 298,002 types (656,334 documents, 28,232,774 tokens) to 239,782 types (656,334 documents, 26,564,509 tokens)
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
##  ......iteration 1 elapsed time: 5.99 seconds (alpha: 0.0446)
##  ......iteration 2 elapsed time: 11.97 seconds (alpha: 0.0390)
##  ......iteration 3 elapsed time: 17.96 seconds (alpha: 0.0335)
##  ......iteration 4 elapsed time: 23.80 seconds (alpha: 0.0281)
##  ......iteration 5 elapsed time: 29.79 seconds (alpha: 0.0225)
##  ......iteration 6 elapsed time: 35.41 seconds (alpha: 0.0176)
##  ......iteration 7 elapsed time: 39.89 seconds (alpha: 0.0136)
##  ......iteration 8 elapsed time: 43.84 seconds (alpha: 0.0101)
##  ......iteration 9 elapsed time: 47.40 seconds (alpha: 0.0072)
##  ......iteration 10 elapsed time: 50.93 seconds (alpha: 0.0043)
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
## text1 0.1181946 0.2552574
## text2 0.4507970 0.3489597
## text3 0.2558150 0.1595985
## text4 0.5279363 0.6595684
## text5 0.4034797 0.4403722
## text6 0.4882109 0.5838685
```
