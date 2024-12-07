
# Wordvector: creating word and document vectors

The **wordvector** package is developed to create word and document
vectors using **quanteda**. This package currently supports word2vec
([Mikolov et al., 2013](http://arxiv.org/abs/1310.4546)) and latent
semantic analysis ([Deerwester et al.,
1990](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9)).

## How to install

**wordvector** is currently available only on Github.

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
wdv <- word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)
## Training CBOW model with 50 dimensions
##  ...using 16 threads for distributed computing
##  ...initializing
##  ...negative sampling in 10 iterations
##  ......iteration 1 elapsed time: 6.26 seconds (alpha: 0.0468)
##  ......iteration 2 elapsed time: 12.76 seconds (alpha: 0.0435)
##  ......iteration 3 elapsed time: 18.35 seconds (alpha: 0.0403)
##  ......iteration 4 elapsed time: 23.56 seconds (alpha: 0.0370)
##  ......iteration 5 elapsed time: 28.77 seconds (alpha: 0.0338)
##  ......iteration 6 elapsed time: 33.97 seconds (alpha: 0.0308)
##  ......iteration 7 elapsed time: 40.31 seconds (alpha: 0.0276)
##  ......iteration 8 elapsed time: 45.91 seconds (alpha: 0.0245)
##  ......iteration 9 elapsed time: 51.79 seconds (alpha: 0.0212)
##  ......iteration 10 elapsed time: 57.45 seconds (alpha: 0.0180)
##  ...normalizing vectors
##  ...complete
```

### Similarity between word vectors

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), mode = "word"), n = 10)
##       amazon       forests       obama            america          
##  [1,] "amazon"     "forests"     "obama"          "america"        
##  [2,] "rainforest" "wetlands"    "barack"         "africa"         
##  [3,] "emerald"    "rainforest"  "biden"          "american"       
##  [4,] "yasuni"     "forest"      "kerry"          "dakota"         
##  [5,] "tinder"     "rainforests" "administration" "carolina"       
##  [6,] "patagonia"  "herds"       "hagel"          "americas"       
##  [7,] "hectare"    "habitat"     "boehner"        "america-focused"
##  [8,] "re-grown"   "farmland"    "rodham"         "korea"          
##  [9,] "franchisee" "temperate"   "karzai"         "koreans"        
## [10,] "sequoia"    "mangrove"    "netanyahu"      "carolina-based" 
##       afghanistan     
##  [1,] "afghanistan"   
##  [2,] "afghan"        
##  [3,] "kabul"         
##  [4,] "taliban"       
##  [5,] "pakistan"      
##  [6,] "afghans"       
##  [7,] "kandahar"      
##  [8,] "mazar-i-sharif"
##  [9,] "somalia"       
## [10,] "gardez"
```

### Arithmetic operations of word vectors

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
analogy(wdv, ~ amazon - forests) # What is Amazon without forests?
##               word similarity
## 1           gawker  0.5608441
## 2        smash-hit  0.5505689
## 3            iliad  0.5399390
## 4           italia  0.5248575
## 5             sony  0.5195535
## 6  biggest-selling  0.5146165
## 7           telmex  0.5123942
## 8     nbcuniversal  0.5112653
## 9          comcast  0.5101601
## 10       telephony  0.5089706
```

``` r
analogy(wdv, ~ obama - america + afghanistan) # What is for Afghanistan as Obama for America? 
##         word similarity
## 1     karzai  0.7338677
## 2    taliban  0.7002406
## 3     afghan  0.6800769
## 4      hamid  0.6701872
## 5      kabul  0.6423406
## 6   haqqanis  0.6193317
## 7  us-afghan  0.6031197
## 8       nato  0.5924604
## 9    afghans  0.5909458
## 10     nawaz  0.5694955
```

These examples replicates analogical tasks in the original word2vec
paper.

``` r
analogy(wdv, ~ berlin - germany + france) # What is for France as Berlin for Germany?
##          word similarity
## 1       paris  0.8782719
## 2   stockholm  0.7868182
## 3    brussels  0.7664732
## 4   amsterdam  0.7493143
## 5      london  0.7363011
## 6  strasbourg  0.7301854
## 7  copenhagen  0.7240090
## 8    helsinki  0.7150432
## 9      warsaw  0.7083805
## 10    bourget  0.7011284
```

``` r
analogy(wdv, ~ quick - quickly + slowly) # What is for slowly as quick for quickly?
##           word similarity
## 1         slow  0.7049219
## 2       uneven  0.6729745
## 3      sharper  0.6668627
## 4       cheeks  0.6437815
## 5          rut  0.6330770
## 6   unexpected  0.6287393
## 7        abyss  0.6256164
## 8  unstoppable  0.6167072
## 9       fading  0.6160594
## 10         dim  0.6128477
```
