# word2vec

This article shows how to fit
[word2vec](https://arxiv.org/abs/1310.4546) using **wordvector**. Since
the model is trained on **quanteda**’s tokens objects, users can easily
integrate the model in their current work flow.

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
##  ...complete, elapsed time: 94.4 seconds.
## Finished constructing tokens of 656,334 documents.
## Returning tokens of 656,334 documents (45,194,192 tokens, 298,002 types) from tokens_remove()
## Returning tokens of 656,334 documents (45,194,192 tokens, 239,782 types) from tokens_keep()
```

## Train word2vec

[`textmodel_word2vec()`](https://koheiw.github.io/wordvector/reference/textmodel_word2vec.md)
supports both `cbow` (continuous bag-of-words) and `sg` (skip-gram)
models.

``` r

# Set the number of processors
options(wordvector_threads = 16)

# Train word2vec
wov <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)
## Training continuous BOW model with 50 dimensions
##  ...using 16 threads for distributed computing
##  ...initializing
##  ...negative sampling in 10 iterations
##  ......iteration 1 elapsed time: 14.31 seconds (alpha: 0.0460)
##  ......iteration 2 elapsed time: 29.82 seconds (alpha: 0.0414)
##  ......iteration 3 elapsed time: 42.21 seconds (alpha: 0.0379)
##  ......iteration 4 elapsed time: 55.61 seconds (alpha: 0.0341)
##  ......iteration 5 elapsed time: 68.66 seconds (alpha: 0.0301)
##  ......iteration 6 elapsed time: 80.53 seconds (alpha: 0.0266)
##  ......iteration 7 elapsed time: 95.39 seconds (alpha: 0.0222)
##  ......iteration 8 elapsed time: 112.40 seconds (alpha: 0.0173)
##  ......iteration 9 elapsed time: 128.12 seconds (alpha: 0.0128)
##  ......iteration 10 elapsed time: 145.11 seconds (alpha: 0.0078)
##  ...complete
```

``` r

# Extract word vector
dim(as.matrix(wov))
## [1] 79259    50
```

## Find similar words

[`similarity()`](https://koheiw.github.io/wordvector/reference/similarity.md)
is a user-friendly utility function to find similar words. Using the
[`analogy()`](https://koheiw.github.io/wordvector/reference/analogy.md)
wrapper, you can also add or subtract word vectors before finding
similar words.

``` r

head(similarity(wov, "bad"))
##      bad      
## [1,] "bad"    
## [2,] "good"   
## [3,] "trouble"
## [4,] "fatigue"
## [5,] "scary"  
## [6,] "worse"
head(similarity(wov, analogy(~ good - bad)))
##      [,1]        
## [1,] "loyalty"   
## [2,] "reaffirm"  
## [3,] "thank"     
## [4,] "reaffirmed"
## [5,] "achieved"  
## [6,] "courageous"
```

## Predict probability

As a language model, the model allows you to compute the probability of
the word conditional on context words.

``` r

head(probability(wov, "bad", mode = "numeric"))
##                bad
## donald  0.19182473
## trump   0.59198527
## endorse 0.06810645
## short   0.75389056
## period  0.43336250
## time    0.82278664
```
