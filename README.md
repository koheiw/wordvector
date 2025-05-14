
# Wordvector: word and document vector models

The **wordvector** package is developed to create word and document
vectors using **quanteda**. This package currently supports word2vec
([Mikolov et al., 2013](http://arxiv.org/abs/1310.4546)) and latent
semantic analysis ([Deerwester et al.,
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
## Package version: 4.2.1
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
wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)
## Training CBOW model with 50 dimensions
##  ...using 16 threads for distributed computing
##  ...initializing
##  ...negative sampling in 10 iterations
##  ......iteration 1 elapsed time: 4.99 seconds (alpha: 0.0465)
##  ......iteration 2 elapsed time: 10.00 seconds (alpha: 0.0431)
##  ......iteration 3 elapsed time: 14.97 seconds (alpha: 0.0396)
##  ......iteration 4 elapsed time: 19.94 seconds (alpha: 0.0362)
##  ......iteration 5 elapsed time: 25.11 seconds (alpha: 0.0326)
##  ......iteration 6 elapsed time: 30.14 seconds (alpha: 0.0291)
##  ......iteration 7 elapsed time: 35.12 seconds (alpha: 0.0257)
##  ......iteration 8 elapsed time: 40.22 seconds (alpha: 0.0222)
##  ......iteration 9 elapsed time: 45.15 seconds (alpha: 0.0188)
##  ......iteration 10 elapsed time: 50.32 seconds (alpha: 0.0152)
##  ...complete
```

### Similarity between word vectors

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), mode = "word"))
##      amazon        forests       obama            america          
## [1,] "amazon"      "forests"     "obama"          "america"        
## [2,] "rainforest"  "herds"       "barack"         "africa"         
## [3,] "plantations" "rainforests" "biden"          "dakota"         
## [4,] "farms"       "rainforest"  "kerry"          "american"       
## [5,] "patagonia"   "plantations" "administration" "carolina"       
## [6,] "warm-water"  "farmland"    "hagel"          "america-focused"
##      afghanistan  
## [1,] "afghanistan"
## [2,] "afghan"     
## [3,] "kabul"      
## [4,] "pakistan"   
## [5,] "taliban"    
## [6,] "afghans"
```

### Arithmetic operations of word vectors

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
# What is Amazon without forests?
head(similarity(wdv, analogy(~ amazon - forests))) 
##      [,1]         
## [1,] "choo"       
## [2,] "smash-hit"  
## [3,] "yahoo"      
## [4,] "tripadvisor"
## [5,] "univision"  
## [6,] "dreamworks"
```

``` r
# What is for Afghanistan as Obama for America? 
head(similarity(wdv, analogy(~ obama - america + afghanistan))) 
##      [,1]         
## [1,] "taliban"    
## [2,] "afghanistan"
## [3,] "karzai"     
## [4,] "hagel"      
## [5,] "hamid"      
## [6,] "obama"
```

These examples replicates analogical tasks in the original word2vec
paper.

``` r
# What is for France as Berlin for Germany?
head(similarity(wdv, analogy(~ berlin - germany + france))) 
##      [,1]       
## [1,] "paris"    
## [2,] "berlin"   
## [3,] "bourget"  
## [4,] "brussels" 
## [5,] "amsterdam"
## [6,] "france"
```

``` r
# What is for slowly as quick for quickly?
head(similarity(wdv, analogy(~ quick - quickly + slowly)))
##      [,1]             
## [1,] "uneven"         
## [2,] "gravity-defying"
## [3,] "slow"           
## [4,] "super-charged"  
## [5,] "buck"           
## [6,] "sideways"
```
