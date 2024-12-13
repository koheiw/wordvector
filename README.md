
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

The lasted version is available on Github.

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
##  ......iteration 1 elapsed time: 4.18 seconds (alpha: 0.0468)
##  ......iteration 2 elapsed time: 8.28 seconds (alpha: 0.0435)
##  ......iteration 3 elapsed time: 12.49 seconds (alpha: 0.0403)
##  ......iteration 4 elapsed time: 16.65 seconds (alpha: 0.0370)
##  ......iteration 5 elapsed time: 20.95 seconds (alpha: 0.0337)
##  ......iteration 6 elapsed time: 25.10 seconds (alpha: 0.0304)
##  ......iteration 7 elapsed time: 29.22 seconds (alpha: 0.0272)
##  ......iteration 8 elapsed time: 33.49 seconds (alpha: 0.0239)
##  ......iteration 9 elapsed time: 37.66 seconds (alpha: 0.0206)
##  ......iteration 10 elapsed time: 41.92 seconds (alpha: 0.0173)
##  ...normalizing vectors
##  ...complete
```

### Similarity between word vectors

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), mode = "word"), n = 10)
##       amazon        forests         obama            america           
##  [1,] "amazon"      "forests"       "obama"          "america"         
##  [2,] "rainforest"  "herds"         "barack"         "dakota"          
##  [3,] "ranches"     "rainforests"   "biden"          "africa"          
##  [4,] "grassland"   "wetlands"      "hagel"          "american"        
##  [5,] "emerald"     "rainforest"    "kerry"          "america-focused" 
##  [6,] "gorges"      "plantations"   "administration" "carolina"        
##  [7,] "plantations" "deforestation" "clinton"        "rhine-westphalia"
##  [8,] "wetlands"    "forest"        "rodham"         "carolina-based"  
##  [9,] "clams"       "ecosystem"     "cluelessly"     "korea"           
## [10,] "flocks"      "ecosystems"    "overture"       "koreans"         
##       afghanistan  
##  [1,] "afghanistan"
##  [2,] "afghan"     
##  [3,] "kabul"      
##  [4,] "pakistan"   
##  [5,] "taliban"    
##  [6,] "afghans"    
##  [7,] "iraq"       
##  [8,] "kandahar"   
##  [9,] "nato-led"   
## [10,] "gardez"
```

### Arithmetic operations of word vectors

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
analogy(wdv, ~ amazon - forests) # What is Amazon without forests?
##               word similarity
## 1        smash-hit  0.6323895
## 2       activision  0.5847817
## 3           gawker  0.5647243
## 4            edits  0.5542108
## 5         t-mobile  0.5461421
## 6             choo  0.5412209
## 7            iliad  0.5328198
## 8            yahoo  0.5301637
## 9         metropcs  0.5285186
## 10 music-streaming  0.5248313
```

``` r
analogy(wdv, ~ obama - america + afghanistan) # What is for Afghanistan as Obama for America? 
##           word similarity
## 1       karzai  0.7449225
## 2        hamid  0.7083441
## 3      taliban  0.7027562
## 4       afghan  0.6663520
## 5        kabul  0.6441681
## 6     haqqanis  0.6000245
## 7    fazlullah  0.5989802
## 8    islamabad  0.5906746
## 9        hagel  0.5866968
## 10 unannounced  0.5851281
```

These examples replicates analogical tasks in the original word2vec
paper.

``` r
analogy(wdv, ~ berlin - germany + france) # What is for France as Berlin for Germany?
##               word similarity
## 1            paris  0.8952651
## 2        amsterdam  0.7984164
## 3        stockholm  0.7670519
## 4         brussels  0.7610676
## 5       copenhagen  0.7607804
## 6         helsinki  0.7183915
## 7  seyne-les-alpes  0.7146337
## 8       strasbourg  0.7103182
## 9           warsaw  0.7064566
## 10          london  0.6942944
```

``` r
analogy(wdv, ~ quick - quickly + slowly) # What is for slowly as quick for quickly?
##        word similarity
## 1    uneven  0.7272214
## 2      slow  0.7151594
## 3      buck  0.6792641
## 4      doom  0.6347708
## 5  sideways  0.6327578
## 6       dim  0.6239180
## 7   sharper  0.6237087
## 8    cooler  0.6216082
## 9  wretched  0.6169135
## 10  limping  0.6166389
```
