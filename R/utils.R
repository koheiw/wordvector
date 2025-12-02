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

#' Compute similarity between word or document vectors
#' 
#' Compute the cosine similarity between word vectors for selected words.
#' @param x a `textmodel_wordvector` object.
#' @param targets words or documents for which similarity is computed.
#' @param layer the layer based on which similarity is computed. This must be "documents" 
#'   when `targets` are document names.
#' @param mode specify the type of resulting object.
#' @return a `matrix` of cosine similarity scores when `mode = "numeric"` or of 
#'   words sorted in descending order by the similarity scores when `mode = "character"`.
#'   When `words` is a named numeric vector, word (or document) vectors are weighted and summed 
#'   before computing similarity scores.
#' @export
#' @seealso [probability()]
similarity <- function(x, targets, layer = c("words", "documents"),
                       mode = c("character", "numeric")) {
    
    layer <- match.arg(layer)
    mode <- ifelse(mode == "words", "character", mode) # for < v0.6.0
    mode <- ifelse(mode == "values", "numeric", mode) # for < v0.6.0
    mode <- match.arg(mode)
    emb1 <- as.matrix(x, layer = layer, normalize = TRUE)
    
    if (!"textmodel_wordvector" %in% class(x))
        stop("x must be a textmodel_wordvector object")
    
    if (is.character(targets)) {
        targets <- structure(rep(1.0, length(targets)), names = targets)
        weighted <- FALSE
    } else if (is.numeric(targets)) {
        if (is.null(names(targets)))
            stop("targets must be named")
        weighted <- TRUE 
    } else {
        stop("targets must be a character vector or a named numeric vector")
    }
    b <- names(targets) %in% rownames(emb1)
    if (sum(!b) == 1) {
        warning(paste0('"', names(targets[!b]), '"',  collapse = ", "),  ' is not found')
    } else if (sum(!b) > 1) {
        warning(paste0('"', names(targets[!b]), '"',  collapse = ", "),  ' are not found')
    }
    targets <- targets[b]
    if (weighted) {
        emb2 <- rbind(colSums(emb1[names(targets),, drop = FALSE] * targets))
    } else {
        emb2 <- emb1[names(targets),, drop = FALSE]
    }
    res <- as.matrix(proxyC::simil(emb1, emb2, use_nan = TRUE))
    if (ncol(res) == 0) {
        res <- matrix(nrow = 0, ncol = 0)
    } else {
        if (mode == "character") {
            res <- apply(res, 2, function(v) {
                names(sort(v, decreasing = TRUE))
            })
        }
    }
    return(res)
}

#' Compute probability of words
#'
#' Compute the probability of words given other words.
#' @param x a trained `textmodel_wordvector` object.
#' @param targets words for which probabilities are computed.
#' @param layer the layer based on which probabilities are computed.
#' @param mode specify the type of resulting object.
#' @return a matrix of words or documents sorted in descending order by the probability 
#'   scores when `mode = "character"`; a matrix of the probability scores when `mode = "numeric"`.
#'   When `words` is a named numeric vector, probability scores are weighted by
#'   the values.
#' @export
#' @seealso [similarity()]
probability <- function(x, targets, layer = c("words", "documents"),
                        mode = c("character", "numeric")) {
    
    layer <- match.arg(layer)
    mode <- ifelse(mode == "words", "character", mode) # for < v0.6.0
    mode <- ifelse(mode == "values", "numeric", mode) # for < v0.6.0
    mode <- match.arg(mode)
    
    if ("textmodel_word2vec" %in% class(x) && layer == "documents")
        stop("textmodel_word2vec does not have the layer for documents")
    
    if (!"textmodel_wordvector" %in% class(x))
        stop("x must be a textmodel_wordvector object")
    
    if (is.null(x$weights))
        stop("x must be a trained textmodel_wordvector object")
    
    if (x$normalize)
        stop("x must be trained with normalize = FALSE")
    
    if (is.character(targets)) {
        targets <- structure(rep(1.0, length(targets)), names = targets)
        weighted <- FALSE
    } else if (is.numeric(targets)) {
        if (is.null(names(targets)))
            stop("targets must be named")
        weighted <- TRUE 
    } else {
        stop("targets must be a character vector or a named numeric vector")
    }

    b <- names(targets) %in% rownames(x$weights)
    if (sum(!b) == 1) {
        warning(paste0('"', names(targets[!b]), '"',  collapse = ", "),  ' is not found')
    } else if (sum(!b) > 1) {
        warning(paste0('"', names(targets[!b]), '"',  collapse = ", "),  ' are not found')
    }
    targets <- targets[b]
    
    values <- as.matrix(x, layer = layer, normalize = FALSE)
    e <- exp(values %*% t(x$weights[names(targets),, drop = FALSE]))
    prob <- e / (e + 1) # sigmoid function
    
    res <- prob %*% diag(targets)
    colnames(res) <- names(targets)
    if (weighted)
        res <- cbind(rowSums(res))
    if (ncol(res) == 0) {
        res <- matrix(nrow = 0, ncol = 0)
    } else {
        if (mode == "character") {
            res <- apply(res, 2, function(v) {
                names(sort(v, decreasing = TRUE))
            })
        }
    }
    return(res)
}

#' Compute perplexity of a model
#'
#' Compute the perplexity of a trained word2vec model with data.
#' @param x a trained `textmodel_wordvector` object.
#' @param targets words for which probabilities are computed.
#' @param data a [quanteda::tokens] or [quanteda::dfm]; the probabilities of words are 
#'    tested against occurrences of words in it.
#' @export
#' @keywords internal
perplexity <- function(x, targets, data) {
    x <- upgrade_pre06(x)
    
    if (!is.character(targets))
        stop("targets must be a character vector")
    
    if (!is.tokens(data) && !is.dfm(data))
        stop("data must be a tokens or dfm")
    data <- dfm(data, remove_padding = TRUE, tolower = x$tolower)
    
    p <- probability(x, targets, mode = "numeric")
    pred <- dfm_match(dfm_weight(data, "prop"), rownames(p)) %*% p
    tri <- Matrix::mat2triplet(dfm_match(data, colnames(pred)))
    exp(-sum(tri$x * log(pred[cbind(tri$i, tri$j)])) / sum(tri$x))
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

upgrade_pre06 <- function(x) {
    
    if (is.list(x$values))
        return(x)
    if (identical(class(x), "textmodel_wordvector")) {
        x$values <- list(word = x$values)
        class(x) <- c("textmodel_word2vec", "textmodel_wordvector")
    } else if (identical(class(x), "textmodel_docvector")) {
        x$values  <- list(doc = x$values)
        class(x) <- c("textmodel_doc2vec", "textmodel_wordvector")
    }
    if (is.numeric(x$type)) {
        x$type <- c("cbow", "sg")[x$type]
    }
    if (is.null(x$tolower)) {
        x$tolower <- TRUE
    }
    return(x)
}

is_word2vec <- function(x) {
    identical(class(x), c("textmodel_word2vec", "textmodel_wordvector"))
}

is_doc2vec <- function(x) {
    identical(class(x), c("textmodel_doc2vec", "textmodel_wordvector"))
}

check_word2vec <- function(x) {
    if (is_word2vec(x)) {
        return(x)
    } else {
        stop("model must be a trained textmodel_word2vec")
    }
}

check_doc2vec <- function(x) {
    if (is_doc2vec(x)) {
        return(x)
    } else {
        stop("model must be a trained textmodel_doc2vec")
    }
}

check_model <- function(x, allow = c("word2vec", "doc2vec", "lsa")) {
    allow <- match.arg(allow, several.ok = TRUE)
    m <- paste0("textmodel_", allow)
    if (any(class(x)[1] == m & class(x)[2] == "textmodel_wordvector")) {
        return(x)
    } else {
        stop("model must be a trained ", paste(m, collapse = " or "))
    }
}

