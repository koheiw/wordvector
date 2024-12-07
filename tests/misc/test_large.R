library(quanteda)
library(word2vec)
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

wdv <- word2vec(toks, dim = 50, type = "cbow", min_count = 5, verbose = TRUE)

analogy(wdv, ~ france + terror, exclude = FALSE, n = 10)
analogy(wdv, ~ britain, exclude = FALSE, n = 20)
analogy(wdv, ~ america, exclude = FALSE, n = 20)
analogy(wdv, ~ america - obama, exclude = FALSE, n = 10)
analogy(wdv, ~ america - trump, exclude = FALSE, n = 10)
analogy(wdv, ~ eu - immigrants, exclude = FALSE, n = 10)
analogy(wdv, ~ eu - refugees, exclude = FALSE, n = 10)
analogy(wdv, ~ eu - terrorism, exclude = FALSE, n = 10)
