library(quanteda)
library(wordvector)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()

toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = FALSE) %>% 
    tokens_tolower()

wov <- word2vec(toks, dim = 50, iter = 10, min_count = 2, sample = 1)

test_that("analogy works", {
    
    ana1 <- analogy(wov, ~ us, exclude = FALSE)
    expect_true(ana1$word[1] == "us")
    expect_true(ana1$similarity[1] == 1.0)
    expect_identical(attr(ana1, "weight"), 
                     c("us" = 1))
    
    ana2 <- analogy(wov, ~ people - us, exclude = FALSE)
    expect_true(ana2$word[1] != "us")
    expect_true(ana2$similarity[1] < 1.0)
    expect_identical(attr(ana2, "weight"), 
                     c("people" = 1, "us" = -1))
    
    ana3 <- analogy(wov, ~ us, exclude = TRUE)
    expect_true(ana3$word[1] != "us")
    expect_true(ana3$similarity[1] < 1.0)
    expect_identical(attr(ana3, "weight"), 
                     c("us" = 1))
    
    ana4 <- analogy(wov, ~ us, exclude = FALSE, type = "simil")
    expect_true(ana4$word[1] == "us")
    expect_true(ana4$similarity[1] == 1.0)
    expect_identical(attr(ana4, "weight"), 
                     c("us" = 1))
    
    expect_warning(
        analogy(wov, ~ xxxx, exclude = FALSE, type = "simil"),
        '"xxxx" is not found'
    )
    expect_true(
        suppressWarnings(
            is.data.frame(analogy(wov, ~ xxxx, exclude = FALSE, type = "simil"))
        )
    )
    expect_warning(
        analogy(wov, ~ xxxx, exclude = FALSE, type = "word"),
        '"xxxx" is not found'
    )
    expect_true(
        suppressWarnings(
            is.data.frame(analogy(wov, ~ xxxx, exclude = FALSE, type = "word"))
        )
    )
    
    # different formulas
    expect_equal(
        attr(analogy(wov, ~ us, exclude = FALSE), "weight"),
        c("us" = 1.0)
    )
    expect_equal(
        attr(analogy(wov, ~ -us+people, exclude = FALSE), "weight"),
        c("us" = -1.0, "people" = 1.0)
    )
    expect_equal(
        attr(analogy(wov, ~ +people - us, exclude = FALSE), "weight"),
        c("people" = 1.0, "us" = -1.0)
    )
})

test_that("similarity works", {
    
    
    sim1 <- similarity(wov, "us", mode = "simil")
    expect_true(is.matrix(sim1))
    expect_identical(
        dimnames(sim1),
        list(names(wov$frequency), "us")
    )
    
    sim2 <- similarity(wov, c("us", "people"), mode = "simil")
    expect_true(is.matrix(sim2))
    expect_identical(
        dimnames(sim2),
        list(names(wov$frequency), c("us", "people"))
    )
    
    sim3 <- similarity(wov, "us", mode = "word")
    expect_true(is.matrix(sim3))
    expect_identical(
        sim3[1,],
        c("us" = "us")
    )
    expect_identical(
        dim(sim3),
        c(length(wov$frequency), 1L)
    )
    
    sim4 <- similarity(wov, c("us", "people"), mode = "word")
    expect_true(is.matrix(sim4))
    expect_identical(
        sim4[1,],
        c("us" = "us", "people" = "people")
    )
    expect_identical(
        dim(sim4),
        c(length(wov$frequency), 2L)
    )
    expect_warning(
        similarity(wov, c("xx", "yyy", "us"), mode = "simil"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
        is.matrix(similarity(wov, c("xx", "yyy"), mode = "simil"))
        )
    )
    expect_warning(
        similarity(wov, c("xx", "yyy", "us"), mode = "word"),
        '"xx", "yyy" are not found'
    )
    expect_true(
        suppressWarnings(
            is.matrix(similarity(wov, c("xx", "yyy"), mode = "word"))
        )
    )
})

