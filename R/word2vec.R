#' Word2vec model
#' 
#' Train a Word2vec model (Mikolov et al., 2023) <https://arxiv.org/pdf/1310.4546.pdf> in different architectures on a [quanteda::tokens] object.
#' @param x a [quanteda::tokens] object.
#' @param dim the size of the word vectors.
#' @param type the architecture of the model; either "cbow" (continuous back of words) or "skip-gram".
#' @param min_count the minimum frequency of the words. Words less frequent than 
#'   this in `x` are removed before training.
#' @param window the size of the word window. Words within this window are considered 
#'   to be the context of a target word.
#' @param iter the number of iterations in model training.
#' @param alpha the initial learning rate.
#' @param use_ns if `TRUE`, negative sampling is used. Otherwise, hierarchical softmax 
#'   is used.
#' @param ns_size the size of negative samples. Only used when `use_ns = TRUE`.
#' @param sample the rate of sampling of words based on their frequency. Sampling is 
#'   disabled when `sample = 1.0`
#' @param verbose if `TRUE`, print the progress of training.
#' @param ... additional arguments.
#' @returns Returns a fitted textmodel_wordvector with the following elements:
#'   \item{vectors}{a matrix for word vectors.}
#'   \item{dim}{the size of the word vectors.}
#'   \item{type}{the architecture of the model.}
#'   \item{frequency}{the frequency of words in `x`.}
#'   \item{window}{the size of the word window.}
#'   \item{iter}{the number of iterations in model training.}
#'   \item{alpha}{the initial learning rate.}
#'   \item{use_ns}{the use of negative sampling.}
#'   \item{ns_size}{the size of negative samples.}
#'   \item{concatenator}{the concatenator in `x`.}
#'   \item{call}{the command used to execute the function.}
#'   \item{version}{the version of the wordvector package.}
#' @details
#'  User can changed the number of processors used for the parallel computing via
#'  `options(wordvector_threads)`.
#' 
#' @references 
#'   Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). 
#'   Distributed Representations of Words and Phrases and their Compositionality. 
#'   https://arxiv.org/abs/1310.4546.
#' @export
#' @examples
#' \dontrun{
#' library(quanteda)
#' library(wordvector)
#' 
#' # pre-processing
#' corp <- data_corpus_news2014 
#' toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
#'    tokens_remove(stopwords("en", "marimo"), padding = TRUE) %>% 
#'    tokens_select("^[a-zA-Z-]+$", valuetype = "regex", case_insensitive = FALSE,
#'                  padding = TRUE) %>% 
#'    tokens_tolower()
#'
#' # train word2vec
#' w2v <- word2vec(toks, dim = 50, type = "cbow", min_count = 5, sample = 0.001)
#' head(similarity(w2v, c("berlin", "germany", "france"), mode = "word"))
#' analogy(w2v, ~ berlin - germany + france)
#' }
word2vec <- function(x, dim = 50, type = c("cbow", "skip-gram"), 
                     min_count = 5L, window = ifelse(type == "cbow", 5L, 10L), 
                     iter = 10L, alpha = 0.05, use_ns = TRUE, ns_size = 5L, 
                     sample = 0.001, verbose = FALSE, ...) {
    UseMethod("word2vec")
}

#' @import quanteda
#' @useDynLib wordvector
#' @export
word2vec.tokens <- function(x, dim = 50L, type = c("cbow", "skip-gram"), 
                            min_count = 5L, window = ifelse(type == "cbow", 5L, 10L), 
                            iter = 10L, alpha = 0.05, use_ns = TRUE, ns_size = 5L, 
                            sample = 0.001, verbose = FALSE, ..., old = FALSE) {
    
    type <- match.arg(type)
    dim <- check_integer(dim, min = 2)
    min_count <- check_integer(min_count, min = 0)
    window <- check_integer(window, min = 1)
    iter <- check_integer(iter, min = 1)
    use_ns <- check_logical(use_ns)
    ns_size <- check_integer(ns_size, min_len = 1)
    alpha <- check_double(alpha, min = 0)
    sample <- check_double(sample, min = 0)
    verbose <- check_logical(verbose)

    type <- match(type, c("cbow", "skip-gram"))
    if (old)
        type <- type * 10
    
    # NOTE: use tokens_xptr?
    x <- tokens_trim(x, min_termfreq = min_count, termfreq_type = "count")
    result <- cpp_w2v(as.tokens(x), attr(x, "types"), 
                      minWordFreq = min_count,
                      size = dim, window = window,
                      sample = sample, withHS = !use_ns, negative = ns_size, 
                      threads = get_threads(), iterations = iter,
                      alpha = alpha, type = type, verbose = verbose)
    if (!is.null(result$message))
        stop("Failed to train word2vec (", result$message, ")")

    result$concatenator <- meta(x, field = "concatenator", type = "object")
    result$call <- try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    result$version <- utils::packageVersion("wordvector")
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
        prettyNum(nrow(x$vectors), big.mark = ","), " words.",
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
        prettyNum(nrow(x$vectors), big.mark = ","), " documents.",
"\n", sep = "")
}


#' Extract word vectors
#'
#' Extract word vectors from a `textmodel_wordvector` or `textmodel_docvector` object.
#' @param x a `textmodel_wordvector` or `textmodel_docvector` object.
#' @param ... not used
#' @return a matrix that contain the word vectors in rows
#' @export
as.matrix.textmodel_wordvector <- function(x, ...){
    return(x$vectors) 
}

#' @export
as.matrix.textmodel_docvector <- function(x, ...){
    return(x$vectors) 
}

#' Create distributed representation of documents

#' @param x a [quanteda::tokens] object.
#' @param model a textmodel_wordvector object.
#' @param ... passed to `[word2vec]` when `model = NULL`.
#' @export
doc2vec <- function(x, model = NULL, ...) {
    UseMethod("doc2vec")
}

#' @export
#' @method doc2vec tokens
doc2vec.tokens <- function(x, model = NULL, ...) {
    if (is.null(model)) {
        model <- word2vec(x, ...)
    } else {
        if (!identical(class(model), "textmodel_wordvector"))
            stop("The object for 'model' must be a trained textmodel_wordvector")
    }
    result <- model
    wov <- as.matrix(model)
    dfmt <- dfm_match(dfm(x, remove_padding = TRUE), rownames(wov))
    dov <- Matrix::tcrossprod(dfmt, t(wov)) # NOTE: consider using proxyC::prod
    result$vectors <- dov / sqrt(Matrix::rowSums(dov ^ 2) / ncol(dov))
    class(result) <- "textmodel_docvector"
    result$call <- try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    return(result)
}
