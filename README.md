
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
    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>% 
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                  padding = TRUE) %>% 
    tokens_tolower()

# Train word2vec
wdv <- word2vec(toks, dim = 50, type = "cbow", min_count = 10, verbose = TRUE)
## Training CBOW model with 50 dimensions
##  ...using 16 threads for distributed computing
##  ...initializing
##  ...negative sampling in 10 iterations
##  ......iteration 1 elapsed time: 5.12 seconds (alpha: 0.0460)
##  ......iteration 2 elapsed time: 10.56 seconds (alpha: 0.0417)
##  ......iteration 3 elapsed time: 15.93 seconds (alpha: 0.0375)
##  ......iteration 4 elapsed time: 21.23 seconds (alpha: 0.0334)
##  ......iteration 5 elapsed time: 26.85 seconds (alpha: 0.0291)
##  ......iteration 6 elapsed time: 32.05 seconds (alpha: 0.0252)
##  ......iteration 7 elapsed time: 37.24 seconds (alpha: 0.0211)
##  ......iteration 8 elapsed time: 42.41 seconds (alpha: 0.0172)
##  ......iteration 9 elapsed time: 47.67 seconds (alpha: 0.0131)
##  ......iteration 10 elapsed time: 52.82 seconds (alpha: 0.0091)
##  ...normalizing vectors
##  ...complete
```

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("good", "bad"), mode = "word"), n = 10)
##       good        bad          
##  [1,] "good"      "bad"        
##  [2,] "better"    "good"       
##  [3,] "honest"    "trouble"    
##  [4,] "really"    "fatigue"    
##  [5,] "excellent" "worse"      
##  [6,] "bad"       "scary"      
##  [7,] "obviously" "difficult"  
##  [8,] "happy"     "hard"       
##  [9,] "fantastic" "susceptible"
## [10,] "quite"     "crazy"
```

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), mode = "word"), n = 10)
##       amazon        forests       obama            america           
##  [1,] "amazon"      "forests"     "obama"          "america"         
##  [2,] "patagonia"   "herds"       "kerry"          "africa"          
##  [3,] "ranching"    "rainforests" "biden"          "dakota"          
##  [4,] "rainforest"  "rainforest"  "hagel"          "american"        
##  [5,] "pulp"        "grasslands"  "barack"         "carolina"        
##  [6,] "cove"        "wetlands"    "clinton"        "korea"           
##  [7,] "tahoe"       "forest"      "administration" "carolina-based"  
##  [8,] "ranches"     "wetland"     "calibrated"     "koreans"         
##  [9,] "plantations" "plantations" "rodham"         "rhine-westphalia"
## [10,] "starbucks"   "farmland"    "rouhani"        "palmerston"      
##       afghanistan  
##  [1,] "afghanistan"
##  [2,] "afghan"     
##  [3,] "kabul"      
##  [4,] "taliban"    
##  [5,] "afghans"    
##  [6,] "pakistan"   
##  [7,] "kandahar"   
##  [8,] "iraq"       
##  [9,] "nato-led"   
## [10,] "somalia"
```

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
analogy(wdv, ~ amazon - forests) # What is Amazon without forests?
##            word similarity
## 1          veja  0.6390142
## 2     smash-hit  0.6298606
## 3        gawker  0.6036518
## 4     telephony  0.5979981
## 5     univision  0.5936611
## 6        gossip  0.5865837
## 7         yahoo  0.5855137
## 8      quebecor  0.5814416
## 9      milliyet  0.5794146
## 10 nbcuniversal  0.5732138
```

``` r
analogy(wdv, ~ obama - america + afghanistan) # What is for Afghanistan as Obama for America? 
##           word similarity
## 1      taliban  0.6564511
## 2       afghan  0.6532658
## 3        hamid  0.6474848
## 4       karzai  0.6370347
## 5         nato  0.6047957
## 6        nawaz  0.5936807
## 7        kabul  0.5810252
## 8    us-afghan  0.5751190
## 9        hagel  0.5628388
## 10 unannounced  0.5604019
```

These examples replicates analogical tasks in Mikolov et al., (2013).

``` r
analogy(wdv, ~ berlin - germany + france) # What is for France as Berlin for Germany?
##          word similarity
## 1       paris  0.9142038
## 2    brussels  0.7575926
## 3      kourou  0.7393114
## 4     bourget  0.7249614
## 5  strasbourg  0.7238158
## 6   amsterdam  0.7141502
## 7        rome  0.7115589
## 8      london  0.7037871
## 9    toulouse  0.6704410
## 10     madrid  0.6694752
```

``` r
analogy(wdv, ~ quick - quickly + slowly) # What is for slowly as quick for quickly?
##             word similarity
## 1         uneven  0.7186023
## 2           slow  0.7090036
## 3        limping  0.6854985
## 4       sideways  0.6491090
## 5      backwards  0.6331264
## 6       recovery  0.6311330
## 7           mode  0.6304995
## 8         steady  0.6285782
## 9  export-driven  0.6139635
## 10         bumps  0.6118245
```
