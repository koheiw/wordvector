
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
## Package version: 4.1.0
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
                  padding = TRUE) %>% 
    tokens_tolower()

# Train word2vec
wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)
## Training CBOW model with 50 dimensions
##  ...using 16 threads for distributed computing
##  ...initializing
##  ...negative sampling in 10 iterations
##  ......iteration 1 elapsed time: 3.69 seconds (alpha: 0.0470)
##  ......iteration 2 elapsed time: 7.58 seconds (alpha: 0.0440)
##  ......iteration 3 elapsed time: 11.33 seconds (alpha: 0.0409)
##  ......iteration 4 elapsed time: 15.14 seconds (alpha: 0.0379)
##  ......iteration 5 elapsed time: 18.98 seconds (alpha: 0.0349)
##  ......iteration 6 elapsed time: 22.79 seconds (alpha: 0.0319)
##  ......iteration 7 elapsed time: 26.62 seconds (alpha: 0.0288)
##  ......iteration 8 elapsed time: 30.41 seconds (alpha: 0.0258)
##  ......iteration 9 elapsed time: 34.29 seconds (alpha: 0.0226)
##  ......iteration 10 elapsed time: 38.19 seconds (alpha: 0.0195)
##  ...normalizing vectors
##  ...complete
```

### Similarity between word vectors

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), mode = "word"))
##      amazon       forests         obama    america    afghanistan  
## [1,] "amazon"     "forests"       "obama"  "america"  "afghanistan"
## [2,] "rainforest" "rainforest"    "barack" "africa"   "afghan"     
## [3,] "gorges"     "herds"         "hagel"  "american" "kabul"      
## [4,] "ranches"    "wetlands"      "rodham" "dakota"   "taliban"    
## [5,] "ranching"   "farmland"      "kerry"  "americas" "pakistan"   
## [6,] "re-grown"   "deforestation" "biden"  "carolina" "afghans"
```

### Arithmetic operations of word vectors

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
# What is Amazon without forests?
head(similarity(wdv, analogy(~ amazon - forests))) 
##      [,1]          
## [1,] "smash-hit"   
## [2,] "pbs"         
## [3,] "telephony"   
## [4,] "nbcuniversal"
## [5,] "univision"   
## [6,] "iliad"
```

``` r
# What is for Afghanistan as Obama for America? 
head(similarity(wdv, analogy(~ obama - america + afghanistan))) 
##      [,1]         
## [1,] "karzai"     
## [2,] "taliban"    
## [3,] "hamid"      
## [4,] "afghanistan"
## [5,] "obama"      
## [6,] "afghan"
```

These examples replicates analogical tasks in the original word2vec
paper.

``` r
# What is for France as Berlin for Germany?
head(similarity(wdv, analogy(~ berlin - germany + france))) 
##      [,1]        
## [1,] "paris"     
## [2,] "berlin"    
## [3,] "amsterdam" 
## [4,] "brussels"  
## [5,] "copenhagen"
## [6,] "stockholm"
```

``` r
# What is for slowly as quick for quickly?
head(similarity(wdv, analogy(~ quick - quickly + slowly)))
##      [,1]         
## [1,] "slow"       
## [2,] "sideways"   
## [3,] "slowly"     
## [4,] "uneven"     
## [5,] "unstoppable"
## [6,] "quick"
```
