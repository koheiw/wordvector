
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
##  ......iteration 1 elapsed time: 6.02 seconds (alpha: 0.0466)
##  ......iteration 2 elapsed time: 11.69 seconds (alpha: 0.0432)
##  ......iteration 3 elapsed time: 17.57 seconds (alpha: 0.0397)
##  ......iteration 4 elapsed time: 23.79 seconds (alpha: 0.0361)
##  ......iteration 5 elapsed time: 29.95 seconds (alpha: 0.0324)
##  ......iteration 6 elapsed time: 35.98 seconds (alpha: 0.0289)
##  ......iteration 7 elapsed time: 42.25 seconds (alpha: 0.0253)
##  ......iteration 8 elapsed time: 48.30 seconds (alpha: 0.0217)
##  ......iteration 9 elapsed time: 55.02 seconds (alpha: 0.0182)
##  ......iteration 10 elapsed time: 61.45 seconds (alpha: 0.0148)
##  ...normalizing vectors
##  ...complete
```

### Similarity between word vectors

`similarity()` computes cosine similarity between word vectors.

``` r
head(similarity(wdv, c("amazon", "forests", "obama", "america", "afghanistan"), mode = "word"), n = 10)
##       amazon        forests       obama            america          
##  [1,] "amazon"      "forests"     "obama"          "america"        
##  [2,] "rainforest"  "rainforests" "barack"         "american"       
##  [3,] "peatlands"   "wetlands"    "biden"          "africa"         
##  [4,] "prospectors" "rainforest"  "kerry"          "america-focused"
##  [5,] "plantations" "freshwater"  "hagel"          "dakota"         
##  [6,] "flora"       "herds"       "clinton"        "carolina"       
##  [7,] "gorges"      "habitat"     "administration" "korea"          
##  [8,] "ranches"     "forest"      "karzai"         "afterthought"   
##  [9,] "pollute"     "mangrove"    "calibrated"     "americas"       
## [10,] "castellon"   "farming"     "rodham"         "african"        
##       afghanistan  
##  [1,] "afghanistan"
##  [2,] "afghan"     
##  [3,] "taliban"    
##  [4,] "kabul"      
##  [5,] "pakistan"   
##  [6,] "afghans"    
##  [7,] "kandahar"   
##  [8,] "nato-led"   
##  [9,] "bagram"     
## [10,] "somalia"
```

### Arithmetic operations of word vectors

`analogy()` offers interface for arithmetic operations of word vectors.

``` r
analogy(wdv, ~ amazon - forests) # What is Amazon without forests?
##            word similarity
## 1    activision  0.6042580
## 2  qatari-owned  0.5917209
## 3        gawker  0.5865727
## 4          veja  0.5860218
## 5         edits  0.5841882
## 6           pbs  0.5817677
## 7     prachatai  0.5798839
## 8    huffington  0.5695452
## 9  anti-erdogan  0.5606463
## 10          itv  0.5598021
```

``` r
analogy(wdv, ~ obama - america + afghanistan) # What is for Afghanistan as Obama for America? 
##         word similarity
## 1     karzai  0.7656369
## 2      hamid  0.7595729
## 3    taliban  0.7080868
## 4     afghan  0.6708250
## 5      kabul  0.6260649
## 6      nawaz  0.5994556
## 7  fazlullah  0.5932931
## 8      faizi  0.5834492
## 9     ashraf  0.5754832
## 10      nato  0.5735640
```

These examples replicates analogical tasks in the original word2vec
paper.

``` r
analogy(wdv, ~ berlin - germany + france) # What is for France as Berlin for Germany?
##                     word similarity
## 1                  paris  0.8930394
## 2              stockholm  0.7964069
## 3             copenhagen  0.7918124
## 4              amsterdam  0.7843772
## 5               brussels  0.7797833
## 6               helsinki  0.7501384
## 7                 london  0.7451187
## 8             strasbourg  0.7374163
## 9                   rome  0.7089049
## 10 notre-dame-des-landes  0.6882247
```

``` r
analogy(wdv, ~ quick - quickly + slowly) # What is for slowly as quick for quickly?
##         word similarity
## 1       slow  0.7324359
## 2        rut  0.6785012
## 3  backwards  0.6752999
## 4   sideways  0.6560765
## 5   staggers  0.6281671
## 6    sharper  0.6199083
## 7     deeper  0.6166403
## 8     steady  0.6154114
## 9      bumps  0.6098998
## 10       dim  0.6097908
```
