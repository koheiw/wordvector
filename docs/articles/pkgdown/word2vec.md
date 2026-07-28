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
##  ...complete, elapsed time: 93.6 seconds.
## Finished constructing tokens from 656,334 documents
## tokens_remove() changed from 298,565 types (656,334 documents, 45,194,192 tokens) to 298,002 types (656,334 documents, 28,232,774 tokens)
## tokens_keep() changed from 298,002 types (656,334 documents, 28,232,774 tokens) to 239,782 types (656,334 documents, 26,564,509 tokens)
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
##  ......iteration 1 elapsed time: 7.32 seconds (alpha: 0.0434)
##  ......iteration 2 elapsed time: 15.04 seconds (alpha: 0.0366)
##  ......iteration 3 elapsed time: 22.28 seconds (alpha: 0.0298)
##  ......iteration 4 elapsed time: 29.92 seconds (alpha: 0.0230)
##  ......iteration 5 elapsed time: 37.30 seconds (alpha: 0.0163)
##  ......iteration 6 elapsed time: 42.04 seconds (alpha: 0.0124)
##  ......iteration 7 elapsed time: 45.13 seconds (alpha: 0.0099)
##  ......iteration 8 elapsed time: 47.98 seconds (alpha: 0.0076)
##  ......iteration 9 elapsed time: 50.87 seconds (alpha: 0.0053)
##  ......iteration 10 elapsed time: 53.56 seconds (alpha: 0.0032)
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
## [6,] "lot"
head(similarity(wov, analogy(~ good - bad)))
##      [,1]          
## [1,] "courageous"  
## [2,] "thank"       
## [3,] "side-by-side"
## [4,] "unwavering"  
## [5,] "maa"         
## [6,] "reaffirm"
```

## Predict probability

As a language model, the model allows you to compute the probability of
the word conditional on context words.

``` r

head(probability(wov, "bad", mode = "numeric"))
##               bad
## donald  0.6470457
## trump   0.6948727
## endorse 0.1175879
## short   0.7951612
## period  0.4145912
## time    0.9306425
```
