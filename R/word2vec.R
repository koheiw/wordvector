#' Word2vec model
#' 
#' Train a Word2vec model (Mikolov et al., 2023) <https://arxiv.org/pdf/1310.4546.pdf> in different architectures on a [quanteda::tokens] object.
#' @param x a [quanteda::tokens] object.
#' @param dim size of the word vectors.
#' @param type the architecture of the model; either "cbow" (continuous back of words) or "skip-gram".
#' @param min_count the minimum frequency of the words. Words less frequent than this in the `tokens` object are removed before training.
#' @param window the size of the word window. Words within this window are considered to be the context of a target word.
#' @param iter the number of iterations in training the model.
#' @param alpha the initial learning rate.
#' @param use_ns if `TRUE`, negative sampling is used. Otherwise, hierarchical softmax is used.
#' @param ns_size the size of negative samples. Only used when `use_ns = TRUE`.
#' @param sample the rate of sampling of words based on theri frequency. Sampling is disabled when `sample = 1.0`
#' @param ... additional arguments.
#' @returns Returns a fitted textmodel_wordvector object with the following
#'   elements: \item{model}{a matrix that records the association between
#'   classes and features.}
#'   \item{data}{the original input of `x`.}
#'   \item{feature}{the feature set in `x`}
#'   \item{class}{the class labels in `y`.}
#'   \item{concatenator}{the concatenator in `x`.}
#'   \item{entropy}{the scheme to compute entropy weights.}
#'   \item{boolean}{the use of the Boolean transformation of `x`.}
#'   \item{call}{the command used to execute the function.}
#'   \item{version}{the version of the wordmap package.}#' }
#' @references \url{https://github.com/maxoodf/word2vec}, \url{https://arxiv.org/pdf/1310.4546.pdf}
#' @export
#' @useDynLib wordvector
word2vec <- function(x, dim = 50, type = c("cbow", "skip-gram"), 
                     min_count = 5L, window = ifelse(type == "cbow", 5L, 10L), 
                     iter = 10L, alpha = 0.05, use_ns = TRUE, ns_size = 5L, 
                     sample = 0.001, verbose = FALSE, ...) {
    UseMethod("word2vec")
}

#' @inherit word2vec title description params details seealso return references
#' @export
#' @importFrom quanteda tokens_trim check_integer check_numeric
word2vec.tokens <- function(x, dim = 50L, type = c("cbow", "skip-gram"), 
                            min_count = 5L, window = ifelse(type == "cbow", 5L, 10L), 
                            iter = 10L, use_ns = FALSE, ns_size = 5L,  alpha = 0.05, 
                            sample = 0.001, verbose = FALSE, ..., old = FALSE) {
    
    type <- match.arg(type)
    dim <- check_integer(dim, min = 2)
    type <- match(type, c("cbow", "skip-gram"))
    min_count <- check_integer(min_count, min = 0)
    window <- check_integer(window, min = 1)
    iter <- check_integer(iter, min = 1)
    use_ns <- check_logical(use_ns)
    ns_size <- check_integer(ns_size, min_len = 1)
    alpha <- check_double(alpha, min = 0)
    sample <- check_double(sample, min = 0)
    verbose <- check_logical(verbose)

    if (old)
        type <- type * 10
    
    # NOTE: use tokens_xptr?
    x <- tokens_trim(x, min_termfreq = min_count, termfreq_type = "count")
    result <- cpp_w2v(as.tokens(x), words = attr(x, "types"), 
                      minWordFreq = min_count,
                      size = dim, window = window,
                      sample = sample, withHS = !use_ns, negative = ns_size, 
                      threads = get_threads(), iterations = iter,
                      alpha = alpha, type = type, verbose = verbose)
    if (!is.null(result$message))
        stop("Failed to train word2vec (", result$message, ")")

    result$concatenator <- meta(x, field = "concatenator", type = "object")
    result$call <- try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    return(result)
}

#' Print method for trained word vectors
#' @param x for print method, the object to be printed
#' @param ... unused
#' @method print textmodel_wordvector
#' @keywords internal
#' @export
print.textmodel_wordvector <- function(x, ...) {
    cat("\nCall:\n")
    print(x$call)
    cat("\n", prettyNum(x$dim, big.mark = ","), " dimensions; ",
        prettyNum(nrow(x$model), big.mark = ","), " words.",
        "\n", sep = "")
}

#' Print method for trained document vectors
#' @param x for print method, the object to be printed
#' @param ... unused
#' @method print textmodel_docvector
#' @keywords internal
#' @export
print.textmodel_docvector <- function(x, ...) {
    cat("\nCall:\n")
    print(x$call)
    cat("\n", prettyNum(x$dim, big.mark = ","), " dimensions; ",
        prettyNum(nrow(x$model), big.mark = ","), " documents.",
"\n", sep = "")
}


#' @title Get the word vectors of a word2vec model
#' @description Get the word vectors of a word2vec model as a dense matrix.
#' @param x a word2vec model as returned by \code{\link{word2vec}} or \code{\link{read.word2vec}}
#' @param ... not used
#' @return a matrix with the word vectors where the rownames are the words from the model vocabulary
#' @export
#' @seealso \code{\link{word2vec}}, \code{\link{read.word2vec}}
#' @examples 
#' path  <- system.file(package = "word2vec", "models", "example.bin")
#' model <- read.word2vec(path)
#' 
#' embedding <- as.matrix(model)
as.matrix.textmodel_wordvector <- function(x, ...){
    return(x$model) 
}

#' @export
as.matrix.textmodel_docvector <- function(x, ...){
    return(x$model) 
}

#' Create distributed representation of documents
#' @export
doc2vec <- function(...) {
    UseMethod("doc2vec")
}

#' @export
#' @method doc2vec tokens
doc2vec.tokens <- function(x, model = NULL, ...) {
    if (is.null(model)) {
        result <- word2vec(x, ...)
    } else {
        if (!identical(class(model), "textmodel_wordvector"))
            stop("The object for 'model' must be a trained textmodel_wordvector")
        result <- model
    }
    wov <- as.matrix(result)
    dfmt <- dfm_match(dfm(x, remove_padding = TRUE), rownames(wov))
    dov <- Matrix::tcrossprod(dfmt, t(wov)) # NOTE: consider using proxyC::prod
    result$model <- dov / sqrt(Matrix::rowSums(dov ^ 2) / ncol(dov))
    class(result) <- "textmodel_docvector"
    result$call <- try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    return(result)
}

#' Find similar words via formula interface
#' @export
#' @importFrom utils head tail
analogy <- function(model, formula, n = 10, method = c("cosine", "dot")) {
    
    n <- check_integer(n, min_len = 0)
    method <- match.arg(method)
    
    emb <- t(as.matrix(model))
    if (!identical(class(formula), "formula"))
        stop("The object for 'formula' should be a formula")
    
    f <- tail(as.character(formula), 1)
    match <- stringi::stri_match_all_regex(f, "([+-] )?([\\w]+)")[[1]]
    match[,2] <- stringi::stri_trim(match[,2])
    match[,2][is.na(match[,2])] <- "+"
    weight <- numeric()
    for (i in seq_len(nrow(match))) {
        m <- match[i,]
        if (!m[3] %in% colnames(emb)) {
            warning('"', m[3],  '" is not found')
            next
        }
        if (m[2] == "-") {
            weight <- c(weight, structure(-1.0, names = m[3]))
        } else if (m[2] == "+") {
            weight <- c(weight, structure(1.0, names = m[3]))
        }
    }
    
    v <- emb[,names(weight), drop = FALSE] %*% weight
    if (method == "dot") {
        suppressWarnings({
            s <- rowSums(sqrt(crossprod(emb, v) / nrow(emb)))
        })
    } else {
        # NOTE: consider exposing the method argument
        s <- Matrix::rowSums(proxyC::simil(emb, v, margin = 2, use_nan = TRUE))
    }
    s <- head(sort(s, decreasing = TRUE), n)
    res <- data.frame(word = names(s), 
                      similarity = s, row.names = NULL)
    attr(res, "formula") <- formula
    attr(res, "weight") <- weight
    return(res)
}


#' Find similar words via a vector
#' @export
synonyms <- function(model, terms, n = 10) { # NOTE: consider changing to neighbors()
    terms <- check_character(terms, max_len = Inf)
    n <- check_integer(n, min_len = 0)
    emb <- as.matrix(model)
    terms <- intersect(terms, rownames(emb))
    sim <- proxyC::simil(emb[terms,,drop = FALSE], emb)
    sapply(terms, function(term) {
        head(names(sort(Matrix::colSums(sim[term,,drop = FALSE]), decreasing = TRUE)), n)
    })
}
