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
##  ...complete, elapsed time: 94.1 seconds.
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
##  ......iteration 1 elapsed time: 16.59 seconds (alpha: 0.0453)
##  ......iteration 2 elapsed time: 33.72 seconds (alpha: 0.0404)
##  ......iteration 3 elapsed time: 51.11 seconds (alpha: 0.0355)
##  ......iteration 4 elapsed time: 68.89 seconds (alpha: 0.0305)
##  ......iteration 5 elapsed time: 86.42 seconds (alpha: 0.0256)
##  ......iteration 6 elapsed time: 101.35 seconds (alpha: 0.0214)
##  ......iteration 7 elapsed time: 117.14 seconds (alpha: 0.0170)
##  ......iteration 8 elapsed time: 133.01 seconds (alpha: 0.0126)
##  ......iteration 9 elapsed time: 149.26 seconds (alpha: 0.0081)
##  ......iteration 10 elapsed time: 164.39 seconds (alpha: 0.0039)
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
## [6,] "hard"
head(similarity(wov, analogy(~ good - bad)))
##      [,1]        
## [1,] "thank"     
## [2,] "courageous"
## [3,] "reaffirmed"
## [4,] "achieved"  
## [5,] "thanking"  
## [6,] "reaffirm"
```

## Predict probability

As a language model, the model allows you to compute the probability of
the word conditional on context words.

``` r

head(probability(wov, "bad", mode = "numeric"))
##                bad
## donald  0.62221457
## trump   0.60169161
## endorse 0.08310225
## short   0.84825842
## period  0.52220572
## time    0.75542530
```
