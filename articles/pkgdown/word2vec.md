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
## Package version: 4.4
## Unicode version: 15.1
## ICU version: 74.2
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
##  ...complete, elapsed time: 85.7 seconds.
## Finished constructing tokens from 656,334 documents
## tokens_remove() changed from 45,194,192 tokens (656,334 documents) to 28,232,774 tokens (656,334 documents)
## tokens_keep() changed from 28,232,774 tokens (656,334 documents) to 26,564,509 tokens (656,334 documents)
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
##  ......iteration 1 elapsed time: 17.91 seconds (alpha: 0.0454)
##  ......iteration 2 elapsed time: 33.07 seconds (alpha: 0.0416)
##  ......iteration 3 elapsed time: 48.70 seconds (alpha: 0.0376)
##  ......iteration 4 elapsed time: 65.93 seconds (alpha: 0.0331)
##  ......iteration 5 elapsed time: 85.33 seconds (alpha: 0.0281)
##  ......iteration 6 elapsed time: 102.71 seconds (alpha: 0.0237)
##  ......iteration 7 elapsed time: 120.52 seconds (alpha: 0.0191)
##  ......iteration 8 elapsed time: 138.86 seconds (alpha: 0.0144)
##  ......iteration 9 elapsed time: 156.38 seconds (alpha: 0.0099)
##  ......iteration 10 elapsed time: 174.70 seconds (alpha: 0.0052)
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
## [3,] "bumps"  
## [4,] "hard"   
## [5,] "scary"  
## [6,] "trouble"
head(similarity(wov, analogy(~ good - bad)))
##      [,1]            
## [1,] "courageous"    
## [2,] "loyalty"       
## [3,] "thank"         
## [4,] "thanking"      
## [5,] "achieved"      
## [6,] "ill-considered"
```

## Predict probability

As a language model, the model allows you to compute the probability of
the word conditional on context words.

``` r

head(probability(wov, "bad", mode = "numeric"))
##               bad
## donald  0.7209594
## trump   0.6459458
## endorse 0.1184928
## short   0.7269336
## period  0.5666892
## time    0.8689582
```
