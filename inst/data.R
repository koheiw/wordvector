library(quanteda)

dat <- readRDS('~/yahoo-news.RDS')
dat$text <- paste0(dat$head, ". ", dat$body)
dat$body <- NULL
corp <- corpus(dat, text_field = 'text')

set.seed(1234)
data_corpus_news2014 <- corpus_sample(corp, 20000)

usethis::use_data(data_corpus_news2014)
