library(quanteda)
library(wordvector)
library(quanteda.textstats)
options(wordvector_threads = 8)
quanteda_options(verbose = TRUE)

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

wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, iter = 10, alpha = 0.1, verbose = TRUE)
similarity(wdv, analogy(~ washington - america + france)) %>% 
    head()

wdv2 <- textmodel_word2vec(toks, dim = 50, type = "sg", min_count = 5, iter = 10, alpha = 0.1, verbose = TRUE)
similarity(wdv2, analogy(~ washington - america + france)) %>% 
    head()

for (i in 1:10) {
    cat(i, "\n")
    wdv <- textmotel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)
}

similarity(wdv, analogy(~ washington - america + france)) %>% 
    head()
similarity(wdv, analogy(~ berlin - germany + france)) %>% 
    head()
