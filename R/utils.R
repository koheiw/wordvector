#' Convert formula to named character vector
#' 
#' Convert a formula to a named character vector in analogy tasks.
#' @param formula a [formula] object that defines the relationship between words 
#'   using `+` or `-` operators.
#' @export
#' @seealso [similarity()]
#' @importFrom utils head tail
#' @return a named character vector to be passed to [similarity()].
#' @examples
#' analogy(~ berlin - germany + france)
#' analogy(~ quick - quickly + slowly)
analogy <- function(formula) {
    
    if (!identical(class(formula), "formula"))
        stop("formula must be a formula object")
    
    f <- tail(as.character(formula), 1)
    match <- stringi::stri_match_all_regex(f, "([+-])?\\s*(\\w+)")[[1]]
    match[,2] <- stringi::stri_trim(match[,2])
    match[,2][is.na(match[,2])] <- "+"
    res <- numeric()
    for (i in seq_len(nrow(match))) {
        m <- match[i,]
        if (m[2] == "-") {
            res <- c(res, structure(-1.0, names = m[3]))
        } else if (m[2] == "+") {
            res <- c(res, structure(1.0, names = m[3]))
        }
    }
    return(res)
}

#' Compute similarity between word vectors
#' 
#' Compute cosine similarity between word vectors for selected words.
#' @param x a `textmodel_wordvector` object.
#' @param words words for which similarity is computed.
#' @param mode specify the type of resulting object.
#' @return a `matrix` of cosine similarity scores when `mode = "values"` or of 
#'   words sorted in descending order by the similarity scores when `mode = "words"`.
#' @export
#' @seealso [analogy()]
similarity <- function(x, words, mode = c("words", "values")) {
    
    if (!identical(class(x), "textmodel_wordvector"))
        stop("x must be a textmodel_wordvector object")

    mode <- match.arg(mode)
    emb1 <- as.matrix(x)
    
    if (is.character(words)) {
        words <- structure(rep(1.0, length(words)), names = words)
        weighted <- FALSE
    } else if (is.numeric(words)) {
        if (is.null(names(words)))
            stop("words must be named")
        weighted <- TRUE 
    } else {
        stop("words must be a character or named numeric vector")
    }
    b <- names(words) %in% rownames(emb1)
    if (sum(!b) == 1) {
        warning(paste0('"', names(words[!b]), '"',  collapse = ", "),  ' is not found')
    } else if (sum(!b) > 1) {
        warning(paste0('"', names(words[!b]), '"',  collapse = ", "),  ' are not found')
    }
    words <- words[b]
    if (weighted) {
        emb2 <- rbind(colSums(emb1[names(words),, drop = FALSE] * words))
    } else {
        emb2 <- emb1[names(words),, drop = FALSE]
    }
    res <- as.matrix(proxyC::simil(emb1, emb2, use_nan = TRUE))
    if (ncol(res) == 0) {
        res <- matrix(nrow = 0, ncol = 0)
    } else {
        if (mode == "words") {
            res <- apply(res, 2, function(v) {
                names(sort(v, decreasing = TRUE))
            })
        }
    }
    return(res)
}

#' \[experimental\] Extract word vector weights
#' 
#' @param x a `textmodel_wordvector` object.
#' @param mode specify the type of resulting object.
#' @return a `matrix` of word vector weights when `mode = "value"` or of 
#'   words sorted in descending order by the weights when `mode = "word"`.
#' @export
weight <- function(x, mode = c("words", "values")) {
    
    if (!identical(class(x), "textmodel_wordvector"))
        stop("x must be a textmodel_wordvector object")
    
    mode <- match.arg(mode)
    if (mode == "values") {
        res <- x$weights
    } else {
        res <- apply(x$weights, 2, function(x) {
            names(sort(x, decreasing = TRUE))
        })
    }
    return(res)
}

get_threads <- function() {
    
    # respect other settings
    default <- c("tbb" = as.integer(Sys.getenv("RCPP_PARALLEL_NUM_THREADS")),
                 "omp" = as.integer(Sys.getenv("OMP_THREAD_LIMIT")),
                 "max" = cpp_get_max_thread())
    default <- unname(min(default, na.rm = TRUE))
    suppressWarnings({
        value <- as.integer(getOption("wordvector_threads", default))
    })
    if (length(value) != 1 || is.na(value)) {
        stop("wordvector_threads must be an integer")
    }
    return(value)
}
