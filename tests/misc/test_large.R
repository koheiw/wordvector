library(quanteda)
library(wordvector)
library(quanteda.textstats)
options(wordvector_threads = 8)

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

wdv <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE, iter = 10, alpha = 0.1)
similarity(wdv, analogy(~ washington - america + france)) %>% 
    head()

wdv2 <- textmodel_word2vec(toks, dim = 50, type = "skip-gram2", min_count = 5, verbose = TRUE, iter = 10, alpha = 0.1)
similarity(wdv2, analogy(~ washington - america + france)) %>% 
    head()

sim <- proxyC::simil(
    wdv2$doc_values,
    wdv2$doc_values["4263794",, drop = FALSE]
)
tail(sort(s <- rowSums(sim)))
print(tail(toks[order(s)]), max_ntoken = -1)


for (i in 1:10) {
    cat(i, "\n")
    wdv <- textmotel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)
}

similarity(wdv, analogy(~ washington - america + france)) %>% 
    head()
similarity(wdv, analogy(~ berlin - germany + france)) %>% 
    head()
