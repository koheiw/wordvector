
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
##  ......iteration 1 elapsed time: 3.65 seconds (alpha: 0.0468)
##  ......iteration 2 elapsed time: 7.64 seconds (alpha: 0.0435)
##  ......iteration 3 elapsed time: 11.45 seconds (alpha: 0.0402)
##  ......iteration 4 elapsed time: 15.51 seconds (alpha: 0.0370)
##  ......iteration 5 elapsed time: 19.46 seconds (alpha: 0.0337)
##  ......iteration 6 elapsed time: 23.36 seconds (alpha: 0.0303)
##  ......iteration 7 elapsed time: 27.29 seconds (alpha: 0.0271)
##  ......iteration 8 elapsed time: 31.08 seconds (alpha: 0.0237)
##  ......iteration 9 elapsed time: 35.06 seconds (alpha: 0.0204)
##  ......iteration 10 elapsed time: 39.00 seconds (alpha: 0.0172)
##  ...normalizing vectors
##  ...complete
```

### Similarity between word vectors

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), mode = "word"), n = 10)
##       amazon        forests         obama            america          
##  [1,] "amazon"      "forests"       "obama"          "america"        
##  [2,] "rainforest"  "herds"         "barack"         "africa"         
##  [3,] "peatlands"   "rainforests"   "biden"          "american"       
##  [4,] "rainforests" "wetlands"      "kerry"          "dakota"         
##  [5,] "rangeland"   "rainforest"    "hagel"          "americas"       
##  [6,] "ranches"     "forest"        "administration" "america-focused"
##  [7,] "wetlands"    "farms"         "boehner"        "carolina"       
##  [8,] "tributary"   "valleys"       "rodham"         "african"        
##  [9,] "ranching"    "farming"       "karzai"         "korea"          
## [10,] "groveland"   "deforestation" "rouhani"        "africans"       
##       afghanistan  
##  [1,] "afghanistan"
##  [2,] "afghan"     
##  [3,] "kabul"      
##  [4,] "taliban"    
##  [5,] "afghans"    
##  [6,] "pakistan"   
##  [7,] "kandahar"   
##  [8,] "nato-led"   
##  [9,] "iraq"       
## [10,] "islamabad"
```

### Arithmetic operations of word vectors

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
analogy(wdv, ~ amazon - forests) # What is Amazon without forests?
##           word similarity
## 1    smash-hit  0.6140768
## 2    univision  0.5964522
## 3        iliad  0.5699427
## 4         choo  0.5629618
## 5      comcast  0.5625957
## 6       gawker  0.5599283
## 7    luxottica  0.5573746
## 8  tripadvisor  0.5504397
## 9      directv  0.5482493
## 10        imax  0.5481854
```

``` r
analogy(wdv, ~ obama - america + afghanistan) # What is for Afghanistan as Obama for America? 
##        word similarity
## 1    karzai  0.6180065
## 2   taliban  0.6113779
## 3  al-abadi  0.6027466
## 4    haider  0.6005121
## 5     nawaz  0.5955588
## 6     hamid  0.5885077
## 7   massoum  0.5866054
## 8     abadi  0.5829245
## 9     hagel  0.5762865
## 10 pentagon  0.5701790
```

These examples replicates analogical tasks in the original word2vec
paper.

``` r
analogy(wdv, ~ berlin - germany + france) # What is for France as Berlin for Germany?
##          word similarity
## 1       paris  0.8931474
## 2    brussels  0.7817763
## 3   amsterdam  0.7794727
## 4   stockholm  0.7647855
## 5  copenhagen  0.7624817
## 6    helsinki  0.7350899
## 7      london  0.7260263
## 8      madrid  0.7089564
## 9  strasbourg  0.7065544
## 10     warsaw  0.6907185
```

``` r
analogy(wdv, ~ quick - quickly + slowly) # What is for slowly as quick for quickly?
##            word similarity
## 1          slow  0.7620580
## 2        uneven  0.7413045
## 3      sideways  0.6759325
## 4       sharper  0.6669842
## 5  supercharged  0.6547219
## 6           dim  0.6526230
## 7        steady  0.6509530
## 8         spurt  0.6476913
## 9        upside  0.6471824
## 10      buoyant  0.6438219
```
