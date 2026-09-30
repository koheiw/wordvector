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
## Package version: 4.5.0
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
##  ...complete, elapsed time: 93 seconds.
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
##  ......iteration 1 elapsed time: 13.87 seconds (alpha: 0.0456)
##  ......iteration 2 elapsed time: 28.49 seconds (alpha: 0.0409)
##  ......iteration 3 elapsed time: 42.96 seconds (alpha: 0.0364)
##  ......iteration 4 elapsed time: 56.89 seconds (alpha: 0.0320)
##  ......iteration 5 elapsed time: 72.00 seconds (alpha: 0.0274)
##  ......iteration 6 elapsed time: 86.78 seconds (alpha: 0.0228)
##  ......iteration 7 elapsed time: 99.36 seconds (alpha: 0.0189)
##  ......iteration 8 elapsed time: 115.44 seconds (alpha: 0.0139)
##  ......iteration 9 elapsed time: 131.18 seconds (alpha: 0.0089)
##  ......iteration 10 elapsed time: 145.59 seconds (alpha: 0.0043)
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
## [3,] "fatigue"
## [4,] "trouble"
## [5,] "hard"   
## [6,] "scary"
head(similarity(wov, analogy(~ good - bad)))
##      [,1]            
## [1,] "ill-considered"
## [2,] "mujahideen"    
## [3,] "thank"         
## [4,] "reaffirm"      
## [5,] "courageous"    
## [6,] "reconciled"
```

## Predict probability

As a language model, the model allows you to compute the probability of
the word conditional on context words.

``` r

head(probability(wov, "bad", mode = "numeric"))
##                bad
## donald  0.17446964
## trump   0.47941574
## endorse 0.06359322
## short   0.74004148
## period  0.60904099
## time    0.78985498
```
