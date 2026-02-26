
# Wordvector: word and document vector models

<!-- badges: start -->

[![CRAN
Version](https://www.r-pkg.org/badges/version/wordvector)](https://CRAN.R-project.org/package=wordvector)
[![Downloads](https://cranlogs.r-pkg.org/badges/wordvector)](https://CRAN.R-project.org/package=wordvector)
[![Total
Downloads](https://cranlogs.r-pkg.org/badges/grand-total/wordvector?color=orange)](https://CRAN.R-project.org/package=wordvector)
[![R build
status](https://github.com/koheiw/wordvector/workflows/R-CMD-check/badge.svg)](https://github.com/koheiw/wordvector/actions)
[![codecov](https://codecov.io/gh/koheiw/wordvector/branch/master/graph/badge.svg)](https://app.codecov.io/gh/koheiw/wordvector)

<!-- badges: end -->

The **wordvector** package is developed to create word and document
vectors using **quanteda**. This package currently supports word2vec
([Mikolov et al., 2013](http://arxiv.org/abs/1310.4546)), doc2vec ([Le,
Q. V., & Mikolov, T., 2014](https://doi.org/10.48550/arXiv.1405.4053))
and latent semantic analysis ([Deerwester et al.,
1990](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9)).

## How to install

**wordvector** is available on CRAN.

``` r
install.packages("wordvector")
```

The latest version is available on Github.

``` r
remotes::install_github("koheiw/wordvector")
```

## Example

We train the word2vec model on a [corpus of news summaries collected
from Yahoo
News](https://www.dropbox.com/s/e19kslwhuu9yc2z/yahoo-news.RDS?dl=1) via
RSS between 2012 and 2016.

### Download data

``` r
# download data
download.file('https://www.dropbox.com/s/e19kslwhuu9yc2z/yahoo-news.RDS?dl=1', 
              '~/yahoo-news.RDS', mode = "wb")
```

### Train word2vec

``` r
library(wordvector)
library(quanteda)
## Package version: 4.3.1
## Unicode version: 15.1
## ICU version: 74.1
## Parallel computing: 16 of 16 threads used.
## See https://quanteda.io for tutorials and examples.

# Load data
dat <- readRDS('~/yahoo-news.RDS')
dat$text <- paste0(dat$head, ". ", dat$body)
corp <- corpus(dat, text_field = 'text', docid_field = "tid")

# Pre-processing
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>% 
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                  padding = TRUE)

# Train word2vec
wov <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)
## Training continuous BOW model with 50 dimensions
##  ...using 16 threads for distributed computing
##  ...initializing
##  ...negative sampling in 10 iterations
##  ......iteration 1 elapsed time: 5.91 seconds (alpha: 0.0453)
##  ......iteration 2 elapsed time: 12.12 seconds (alpha: 0.0404)
##  ......iteration 3 elapsed time: 18.30 seconds (alpha: 0.0357)
##  ......iteration 4 elapsed time: 24.58 seconds (alpha: 0.0309)
##  ......iteration 5 elapsed time: 31.38 seconds (alpha: 0.0259)
##  ......iteration 6 elapsed time: 40.15 seconds (alpha: 0.0203)
##  ......iteration 7 elapsed time: 49.13 seconds (alpha: 0.0143)
##  ......iteration 8 elapsed time: 57.64 seconds (alpha: 0.0086)
##  ......iteration 9 elapsed time: 64.40 seconds (alpha: 0.0044)
##  ......iteration 10 elapsed time: 69.44 seconds (alpha: 0.0017)
##  ...complete
```

### Similarity between word vectors

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wov, c("amazon", "forests", "obama", "america", "afghanistan"), 
                mode = "character"))
##      amazon         forests       obama     america    afghanistan  
## [1,] "amazon"       "forests"     "obama"   "america"  "afghanistan"
## [2,] "peatlands"    "herds"       "barack"  "africa"   "afghan"     
## [3,] "rainforest"   "wetlands"    "biden"   "american" "taliban"    
## [4,] "americana"    "rainforests" "kerry"   "dakota"   "kabul"      
## [5,] "ranches"      "rainforest"  "hagel"   "carolina" "pakistan"   
## [6,] "forestethics" "grasslands"  "clinton" "korea"    "iraq"
```

### Arithmetic operations of word vectors

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
# What is Amazon without forests?
head(similarity(wov, analogy(~ amazon - forests))) 
##      [,1]         
## [1,] "smash-hit"  
## [2,] "tripadvisor"
## [3,] "rihanna"    
## [4,] "yahoo"      
## [5,] "pandora"    
## [6,] "univision"
```

``` r
# What is for Afghanistan as Obama for America? 
head(similarity(wov, analogy(~ obama - america + afghanistan))) 
##      [,1]         
## [1,] "afghanistan"
## [2,] "afghan"     
## [3,] "taliban"    
## [4,] "karzai"     
## [5,] "nato"       
## [6,] "hamid"
```

These examples replicates analogical tasks in the original word2vec
paper.

``` r
# What is for France as Berlin for Germany?
head(similarity(wov, analogy(~ berlin - germany + france))) 
##      [,1]        
## [1,] "paris"     
## [2,] "berlin"    
## [3,] "brussels"  
## [4,] "strasbourg"
## [5,] "bourget"   
## [6,] "france"
```

``` r
# What is for slowly as quick for quickly?
head(similarity(wov, analogy(~ quick - quickly + slowly)))
##      [,1]         
## [1,] "slow"       
## [2,] "sideways"   
## [3,] "uneven"     
## [4,] "dim"        
## [5,] "heralds"    
## [6,] "bounce-back"
```
