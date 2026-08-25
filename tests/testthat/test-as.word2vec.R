library(quanteda)
library(wordvector)
options(wordvector_threads = 2)

test_that("as.textmodel_doc2vec works", {
    
    mat <- matrix(rnorm(500), nrow = 5, 
                  dimnames = list(c("a", "b", "c", "d", "e"), NULL))
    wov <- as.textmodel_word2vec(mat)
    
    expect_equal(
        wov$dim, 
        100
    )
    expect_identical(
        wov$values$word, 
        mat
    )
    
    mat2 <- matrix(c(NA, rnorm(499)), nrow = 5, 
                   dimnames = list(c("a", "b", "c", "d", "e"), NULL))
    expect_error(
        as.textmodel_word2vec(mat2),
        "x must be a numeric matrix without NA"
    )
    
    mat3 <- matrix(sample(c(TRUE, FALSE), 500, replace = TRUE), nrow = 5, 
                   dimnames = list(c("a", "b", "c", "d", "e"), NULL))
    expect_error(
        as.textmodel_word2vec(mat3),
        "x must be a numeric matrix without NA"
    )
    
    mat4 <- matrix(rnorm(500), nrow = 5)
    expect_error(
        as.textmodel_word2vec(mat4),
        "x must have rownames for words"
    )
    
})
