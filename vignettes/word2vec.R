## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, cache = TRUE, collapse = TRUE)
if (identical(Sys.getenv("NOT_CRAN"), "false"))
    options(quanteda_threads = 2, wordvector_threads = 2)

## ----eval=TRUE, cache=TRUE----------------------------------------------------
# Load data
f <- tempfile()
download.file('https://www.dropbox.com/s/e19kslwhuu9yc2z/yahoo-news.RDS?dl=1', 
              f, mode = "wb")

## -----------------------------------------------------------------------------
library(quanteda)
library(wordvector)
quanteda_options(verbose = TRUE)

# Construct corpus
dat <- readRDS(f)
dat$text <- paste0(dat$head, ". ", dat$body)
corp <- corpus(dat, text_field = 'text')

# Tokenize
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>% 
    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
                  padding = TRUE)

## ----cache=TRUE---------------------------------------------------------------
# Set the number of processors
options(wordvector_threads = 16)

# Train word2vec
wov <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)

## -----------------------------------------------------------------------------
# Extract word vector
dim(as.matrix(wov))

## -----------------------------------------------------------------------------
head(similarity(wov, "bad"))
head(similarity(wov, analogy(~ good - bad)))

## -----------------------------------------------------------------------------
head(probability(wov, "bad", mode = "numeric"))

