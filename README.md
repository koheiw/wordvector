
# Wordvector: word and document vector models

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
wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)
## Training CBOW model with 50 dimensions
##  ...using 16 threads for distributed computing
##  ...initializing
##  ...negative sampling in 10 iterations
##  ......iteration 1 elapsed time: 6.44 seconds (alpha: 0.0455)
##  ......iteration 2 elapsed time: 13.22 seconds (alpha: 0.0408)
##  ......iteration 3 elapsed time: 19.80 seconds (alpha: 0.0363)
##  ......iteration 4 elapsed time: 26.97 seconds (alpha: 0.0317)
##  ......iteration 5 elapsed time: 34.22 seconds (alpha: 0.0270)
##  ......iteration 6 elapsed time: 41.09 seconds (alpha: 0.0224)
##  ......iteration 7 elapsed time: 47.71 seconds (alpha: 0.0178)
##  ......iteration 8 elapsed time: 54.47 seconds (alpha: 0.0131)
##  ......iteration 9 elapsed time: 61.07 seconds (alpha: 0.0085)
##  ......iteration 10 elapsed time: 67.54 seconds (alpha: 0.0041)
##  ...complete
```

### Similarity between word vectors

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), 
                mode = "character"))
##      amazon       forests       obama                   america          
## [1,] "amazon"     "forests"     "obama"                 "america"        
## [2,] "rainforest" "herds"       "biden"                 "america-focused"
## [3,] "peat"       "rainforests" "relationship-building" "carolina"       
## [4,] "re-grown"   "farmland"    "kerry"                 "american"       
## [5,] "peatlands"  "rainforest"  "hagel"                 "dakota"         
## [6,] "sunflower"  "forest"      "clinton"               "africa"         
##      afghanistan  
## [1,] "afghanistan"
## [2,] "afghan"     
## [3,] "taliban"    
## [4,] "kabul"      
## [5,] "afghans"    
## [6,] "pakistan"
```

### Arithmetic operations of word vectors

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
# What is Amazon without forests?
head(similarity(wdv, analogy(~ amazon - forests))) 
##      [,1]            
## [1,] "yahoo"         
## [2,] "smash-hit"     
## [3,] "gawker"        
## [4,] "aggregators"   
## [5,] "troll"         
## [6,] "globe-spanning"
```

``` r
# What is for Afghanistan as Obama for America? 
head(similarity(wdv, analogy(~ obama - america + afghanistan))) 
##      [,1]         
## [1,] "afghanistan"
## [2,] "karzai"     
## [3,] "afghan"     
## [4,] "taliban"    
## [5,] "obama"      
## [6,] "nato"
```

These examples replicates analogical tasks in the original word2vec
paper.

``` r
# What is for France as Berlin for Germany?
head(similarity(wdv, analogy(~ berlin - germany + france))) 
##      [,1]        
## [1,] "paris"     
## [2,] "strasbourg"
## [3,] "brussels"  
## [4,] "berlin"    
## [5,] "amsterdam" 
## [6,] "france"
```

``` r
# What is for slowly as quick for quickly?
head(similarity(wdv, analogy(~ quick - quickly + slowly)))
##      [,1]       
## [1,] "uneven"   
## [2,] "stumble"  
## [3,] "backwards"
## [4,] "fades"    
## [5,] "slow"     
## [6,] "upside"
```
