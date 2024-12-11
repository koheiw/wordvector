#' \[experimental\] Find analogical relationships between words
#' 
#' @param x a `textmodel_wordvector` object.
#' @param formula a [formula] object that defines the relationship between words 
#'   using `+` or `-` operators.
#' @param n the number of words in the resulting object.
#' @param exclude if `TRUE`, words in `formula` are excluded from the result.
#' @param type specify the type of vectors to be used. "word" is word vectors  
#'   while "simil" is similarity vectors.
#' @return a `data.frame` with the words sorted and their cosine similarity sorted 
#'   in descending order.
#' @importFrom utils head tail
#' @export
#' @references 
#'   Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). 
#'   Distributed Representations of Words and Phrases and their Compositionality. 
#'   http://arxiv.org/abs/1310.4546.
#' @examples
#' \donttest{
#' # from Mikolov et al. (2023)
#' analogy(wdv, ~ berlin - germany + france)
#' analogy(wdv, ~ quick - quickly + slowly)
#' }
analogy <- function(x, formula, n = 10, exclude = TRUE, type = c("word", "simil")) {
    
    if (!identical(class(x), "textmodel_wordvector"))
        stop("x must be a textmodel_wordvector object")
        
    n <- check_integer(n)
    exclude <- check_logical(exclude)
    type <- match.arg(type)
    emb <- as.matrix(x)
    if (!identical(class(formula), "formula"))
        stop("The object for 'formula' should be a formula")
    
    f <- tail(as.character(formula), 1)
    match <- stringi::stri_match_all_regex(f, "([+-])?\\s*(\\w+)")[[1]]
    match[,2] <- stringi::stri_trim(match[,2])
    match[,2][is.na(match[,2])] <- "+"
    weight <- numeric()
    for (i in seq_len(nrow(match))) {
        m <- match[i,]
        if (!m[3] %in% rownames(emb)) {
            warning('"', m[3],  '" is not found')
            next
        }
        if (m[2] == "-") {
            weight <- c(weight, structure(-1.0, names = m[3]))
        } else if (m[2] == "+") {
            weight <- c(weight, structure(1.0, names = m[3]))
        }
    }
    
    j <- match(names(weight), rownames(emb))
    if (length(j) == 0) {
        res <- data.frame(word = character(), similarity = numeric())
    } else {
        v <- emb[j,, drop = FALSE]
        if (exclude)
            emb <- emb[j * -1,, drop = FALSE]
        if (type == "word") {
            s <- Matrix::rowMeans(proxyC::simil(emb, t(t(v) %*% weight), use_nan = TRUE))
        } else {
            s <- Matrix::rowMeans(proxyC::simil(emb, v, use_nan = TRUE) %*% weight)
        }
        s <- head(sort(s, decreasing = TRUE), n)
        res <- data.frame(word = names(s), similarity = s)
    }
    rownames(res) <- NULL
    attr(res, "formula") <- formula
    attr(res, "weight") <- weight
    return(res)
}


#' Compute similarity between word vectors
#' 
#' @param x a `textmodel_wordvector` object.
#' @param words words for which similarity is computed.
#' @param mode specify the type of resulting object.
#' @return a `matrix` of cosine similarity scores when `mode = "simil"` or of 
#'   words sorted by the similarity scores when `mode = "word`.
#' @export
similarity <- function(x, words, mode = c("simil", "word")) {
    
    if (!identical(class(x), "textmodel_wordvector"))
        stop("x must be a textmodel_wordvector object")
    
    words <- check_character(words, max_len = Inf)
    mode <- match.arg(mode)
    emb <- as.matrix(x)
    
    b <- words %in% rownames(emb)
    if (sum(!b) == 1) {
        warning(paste0('"', words[!b], '"',  collapse = ", "),  ' is not found')
    } else if (sum(!b) > 1) {
        warning(paste0('"', words[!b], '"',  collapse = ", "),  ' are not found')
    }
    words <- words[b]
    res <- as.matrix(proxyC::simil(emb, emb[words,, drop = FALSE], use_nan = TRUE))
    if (ncol(res) == 0) {
        res <- matrix(nrow = 0, ncol = 0)
    } else {
        if (mode == "word") {
            res <- apply(res, 2, function(v) {
                names(sort(v, decreasing = TRUE))
            })
        }
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
