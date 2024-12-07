
# Wordvector: word and document vector models

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
##  ......iteration 1 elapsed time: 5.43 seconds (alpha: 0.0466)
##  ......iteration 2 elapsed time: 10.94 seconds (alpha: 0.0429)
##  ......iteration 3 elapsed time: 16.31 seconds (alpha: 0.0395)
##  ......iteration 4 elapsed time: 21.67 seconds (alpha: 0.0359)
##  ......iteration 5 elapsed time: 27.36 seconds (alpha: 0.0322)
##  ......iteration 6 elapsed time: 32.62 seconds (alpha: 0.0288)
##  ......iteration 7 elapsed time: 38.11 seconds (alpha: 0.0253)
##  ......iteration 8 elapsed time: 43.33 seconds (alpha: 0.0219)
##  ......iteration 9 elapsed time: 48.51 seconds (alpha: 0.0185)
##  ......iteration 10 elapsed time: 53.98 seconds (alpha: 0.0149)
##  ...normalizing vectors
##  ...complete
```

### Similarity between word vectors

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), mode = "word"), n = 10)
##       amazon        forests       obama            america           
##  [1,] "amazon"      "forests"     "obama"          "america"         
##  [2,] "rainforest"  "rainforests" "hagel"          "africa"          
##  [3,] "yasuni"      "herds"       "biden"          "american"        
##  [4,] "plantations" "wetlands"    "kerry"          "dakota"          
##  [5,] "gorges"      "ecosystem"   "barack"         "carolina"        
##  [6,] "streams"     "plantations" "rodham"         "america-focused" 
##  [7,] "wetlands"    "rainforest"  "cluelessly"     "rhine-westphalia"
##  [8,] "re-grown"    "forest"      "clinton"        "african"         
##  [9,] "tributary"   "mangroves"   "administration" "carolina-based"  
## [10,] "emerald"     "farming"     "boehner"        "americas"        
##       afghanistan  
##  [1,] "afghanistan"
##  [2,] "afghan"     
##  [3,] "kabul"      
##  [4,] "taliban"    
##  [5,] "afghans"    
##  [6,] "pakistan"   
##  [7,] "iraq"       
##  [8,] "kandahar"   
##  [9,] "bagram"     
## [10,] "somalia"
```

### Arithmetic operations of word vectors

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
analogy(wdv, ~ amazon - forests) # What is Amazon without forests?
##            word similarity
## 1     smash-hit  0.5728512
## 2     univision  0.5455706
## 3         iliad  0.5318353
## 4  nbcuniversal  0.5250018
## 5         edits  0.5223959
## 6     us-europe  0.5156272
## 7        cheesy  0.5155246
## 8     collymore  0.5120331
## 9        airing  0.5038335
## 10      skymark  0.5018828
```

``` r
analogy(wdv, ~ obama - america + afghanistan) # What is for Afghanistan as Obama for America? 
##          word similarity
## 1      karzai  0.7131091
## 2     taliban  0.6573556
## 3       hamid  0.6523524
## 4       hagel  0.6196738
## 5       kabul  0.5950130
## 6      afghan  0.5863461
## 7    pentagon  0.5854797
## 8  commanders  0.5769696
## 9     panetta  0.5755119
## 10      nawaz  0.5634649
```

These examples replicates analogical tasks in the original word2vec
paper.

``` r
analogy(wdv, ~ berlin - germany + france) # What is for France as Berlin for Germany?
##               word similarity
## 1            paris  0.8924838
## 2         brussels  0.7839238
## 3        amsterdam  0.7785592
## 4       copenhagen  0.7760023
## 5       strasbourg  0.7626950
## 6        stockholm  0.7456473
## 7  seyne-les-alpes  0.7248383
## 8         helsinki  0.7115050
## 9             rome  0.7100141
## 10          warsaw  0.6985825
```

``` r
analogy(wdv, ~ quick - quickly + slowly) # What is for slowly as quick for quickly?
##          word similarity
## 1        slow  0.7471051
## 2      upside  0.6909375
## 3   backwards  0.6893690
## 4    sideways  0.6853107
## 5      uneven  0.6823009
## 6         pre  0.6635077
## 7         dim  0.6460451
## 8  unexpected  0.6439453
## 9    brighter  0.6395399
## 10     steady  0.6379962
```
