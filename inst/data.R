library(quanteda)
library(stringi)

dat <- readRDS('~/yahoo-news.RDS')
dat <- dat[order(dat$time),]
dat$head <- stri_replace_all_regex(dat$head, "\\s+", " ") %>% 
    stri_trans_nfkc() %>% 
    stri_trim()
dat$body <- stri_replace_all_regex(dat$body, "\\s+", " ") %>% 
    stri_trans_nfkc() %>% 
    stri_trim()
dat$time <- NULL

# # both head and body
dat$text <- paste0(dat$head, ". ", dat$body)
dat$head <- NULL
dat$body <- NULL
dat <- subset(dat, !duplicated(text))
corp <- corpus(dat, text_field = "text", docid_field = "tid")

# # only body
# dat <- subset(dat, !duplicated(body))
# corp <- corpus(dat, text_field = "body", docid_field = "tid")

# set.seed(1234)
# data_corpus_news <- corpus_sample(corp, 20000)
# usethis::use_data(data_corpus_news)
# 
# set.seed(1234)
# data_corpus_news2015 <- corpus_subset(corp, lubridate::year(date) == 2015) %>%
#     corpus_sample(20000)
# usethis::use_data(data_corpus_news2015)

set.seed(1234)
data_corpus_news2014 <- corpus_subset(corp, lubridate::year(date) == 2014) %>%
    corpus_sample(20000)
usethis::use_data(data_corpus_news2014)
