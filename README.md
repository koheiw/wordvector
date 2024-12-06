
# Wordvector: creating word and cocument vectors

The **wordvector** package is developed to create word and document
vectors using **quanteda**. This package currently supports word2vec
([Mikolov et al., 2013](http://arxiv.org/abs/1310.4546)) and latent
semantic analysis ([Deerwester et al.,
1990](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9)).

## How to install

**wordvector** is currently available only on Github.

``` r
devtools::install_github("koheiw/wordvector")
```

## Example

We train the word2vec model on a [corpus of news summaries collected
from Yahoo
News](https://www.dropbox.com/s/e19kslwhuu9yc2z/yahoo-news.RDS?dl=1) via
RSS in 2014.

``` r
# download data
download.file('https://www.dropbox.com/s/e19kslwhuu9yc2z/yahoo-news.RDS?dl=1', 
              '~/yahoo-news.RDS', mode = "wb")
```

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
dat$body <- NULL
corp <- corpus(dat, text_field = 'text')

# Tokenize
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                  padding = TRUE) %>% 
    tokens_tolower()

# Train word2vec
wdv <- word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)
## Training CBOW model with 50 dimensions
##  ...using 16 threads for distributed computing
##  ...initializing
##  ...negative sampling in 10 iterations
##  ......iteration 1 elapsed time: 5.36 seconds (alpha: 0.0455)
##  ......iteration 2 elapsed time: 10.85 seconds (alpha: 0.0408)
##  ......iteration 3 elapsed time: 16.31 seconds (alpha: 0.0364)
##  ......iteration 4 elapsed time: 21.79 seconds (alpha: 0.0319)
##  ......iteration 5 elapsed time: 27.30 seconds (alpha: 0.0273)
##  ......iteration 6 elapsed time: 32.86 seconds (alpha: 0.0229)
##  ......iteration 7 elapsed time: 38.34 seconds (alpha: 0.0184)
##  ......iteration 8 elapsed time: 43.96 seconds (alpha: 0.0139)
##  ......iteration 9 elapsed time: 49.44 seconds (alpha: 0.0094)
##  ......iteration 10 elapsed time: 54.95 seconds (alpha: 0.0049)
##  ...normalizing vectors
##  ...complete
```

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), mode = "word"), n = 10)
##       amazon       forests       obama        america           afghanistan  
##  [1,] "amazon"     "forests"     "obama"      "america"         "afghanistan"
##  [2,] "acacia"     "herds"       "biden"      "africa"          "afghan"     
##  [3,] "rainforest" "rainforest"  "kerry"      "american"        "kabul"      
##  [4,] "soy"        "grasslands"  "hagel"      "dakota"          "pakistan"   
##  [5,] "fresnillo"  "wetlands"    "unwise"     "america-focused" "taliban"    
##  [6,] "tinder"     "forest"      "clinton"    "korea"           "afghans"    
##  [7,] "ranching"   "rainforests" "cluelessly" "carolina"        "iraq"       
##  [8,] "cerro"      "mangrove"    "rodham"     "palmerston"      "kandahar"   
##  [9,] "patagonia"  "habitats"    "putin"      "carolina-based"  "nato"       
## [10,] "clam"       "plantations" "panetta"    "koreans"         "nato-led"
```

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
analogy(wdv, ~ amazon - forests) # What is Amazon without forests?
##            word similarity
## 1     smash-hit  0.6288743
## 2  nbcuniversal  0.5956835
## 3        gawker  0.5894248
## 4     telephony  0.5832942
## 5         yahoo  0.5789149
## 6  qatari-owned  0.5714233
## 7       verizon  0.5579714
## 8   live-stream  0.5575497
## 9     univision  0.5574111
## 10     langlois  0.5550320
```

``` r
analogy(wdv, ~ obama - america + afghanistan) # What is for Afghanistan as Obama for America? 
##        word similarity
## 1      nato  0.6207850
## 2   taliban  0.5987674
## 3    karzai  0.5980017
## 4    sharif  0.5946231
## 5    afghan  0.5942727
## 6  military  0.5743087
## 7  pentagon  0.5708901
## 8     hagel  0.5641687
## 9     abadi  0.5462650
## 10    kabul  0.5436152
```

These examples replicates analogical tasks in Mikolov et al., (2013).

``` r
analogy(wdv, ~ berlin - germany + france) # What is for France as Berlin for Germany?
##                     word similarity
## 1                  paris  0.9214461
## 2               brussels  0.7645437
## 3              amsterdam  0.6983973
## 4                 london  0.6876289
## 5                bourget  0.6840104
## 6              stockholm  0.6751992
## 7                   rome  0.6750841
## 8                 munich  0.6742921
## 9  notre-dame-des-landes  0.6674540
## 10                madrid  0.6646350
```

``` r
analogy(wdv, ~ quick - quickly + slowly) # What is for slowly as quick for quickly?
##           word similarity
## 1         slow  0.7317324
## 2          rut  0.6925048
## 3       steady  0.6911304
## 4      sharper  0.6810087
## 5       uneven  0.6661730
## 6  unstoppable  0.6572307
## 7     sideways  0.6526448
## 8       wobbly  0.6473020
## 9       fading  0.6441875
## 10  unexpected  0.6435836
```
