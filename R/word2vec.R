#' Word2vec model
#' 
#' Train a Word2vec model (Mikolov et al., 2023) in different architectures on a [quanteda::tokens] object.
#' @param x a [quanteda::tokens] or [quanteda::tokens_xptr] object.
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
#' @param tolower lower-case all the tokens before fitting the model.
#' @param verbose if `TRUE`, print the progress of training.
#' @param ... additional arguments.
#' @returns Returns a textmodel_wordvector object with the following elements:
#'   \item{values}{a matrix for word vector values.}
#'   \item{weights}{a matrix for word vector weights.}
#'   \item{dim}{the size of the word vectors.}
#'   \item{type}{the architecture of the model.}
#'   \item{frequency}{the frequency of words in `x`.}
#'   \item{window}{the size of the word window.}
#'   \item{iter}{the number of iterations in model training.}
#'   \item{alpha}{the initial learning rate.}
#'   \item{use_ns}{the use of negative sampling.}
#'   \item{ns_size}{the size of negative samples.}
#'   \item{min_count}{the value of min_count.}
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
#' \donttest{
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
#' w2v <- textmodel_word2vec(toks, dim = 50, type = "cbow", min_count = 5, sample = 0.001)
#'
#' # find similar words
#' head(similarity(w2v, c("berlin", "germany", "france"), mode = "words"))
#' head(similarity(w2v, c("berlin" = 1, "germany" = -1, "france" = 1), mode = "values"))
#' head(similarity(w2v, analogy(~ berlin - germany + france), mode = "words"))
#' }
textmodel_word2vec <- function(x, dim = 50, type = c("cbow", "skip-gram"), 
                               min_count = 5L, window = ifelse(type == "cbow", 5L, 10L), 
                               iter = 10L, alpha = 0.05, use_ns = TRUE, ns_size = 5L, 
                               sample = 0.001, tolower = TRUE,
                               model = NULL, verbose = FALSE, ...) {
    UseMethod("textmodel_word2vec")
}

#' @import quanteda
#' @useDynLib wordvector
#' @export
#' @method textmodel_word2vec tokens
textmodel_word2vec.tokens <- function(x, dim = 50L, type = c("cbow", "skip-gram"), 
                                      min_count = 5L, window = ifelse(type == "cbow", 5L, 10L), 
                                      iter = 10L, alpha = 0.05, use_ns = TRUE, ns_size = 5L, 
                                      sample = 0.001, normalize = FALSE, tolower = TRUE,
                                      model = NULL, verbose = FALSE, ..., old = FALSE) {
    
    type <- match.arg(type)
    dim <- check_integer(dim, min = 2)
    min_count <- check_integer(min_count, min = 0)
    window <- check_integer(window, min = 1)
    iter <- check_integer(iter, min = 1)
    use_ns <- check_logical(use_ns)
    ns_size <- check_integer(ns_size, min_len = 1)
    alpha <- check_double(alpha, min = 0)
    sample <- check_double(sample, min = 0)
    normalize <- check_logical(normalize)
    tolower <- check_logical(tolower)
    verbose <- check_logical(verbose)
    
    if (normalize)
        .Deprecated(msg = "normalize is deprecated and defaults to FALSE.")
    
    type <- match(type, c("cbow", "skip-gram"))
    if (old)
        type <- type * 10
    
    x <- as.tokens_xptr(x)
    if (tolower)
        x <- tokens_tolower(x)
    x <- tokens_trim(x, min_termfreq = min_count, termfreq_type = "count")
    
    result <- cpp_w2v(x, size = dim, window = window,
                      sample = sample, withHS = !use_ns, negative = ns_size, 
                      threads = get_threads(), iterations = iter,
                      alpha = alpha, type = type, normalize = normalize, model = model,
                      verbose = verbose)
    if (!is.null(result$message))
        stop("Failed to train word2vec (", result$message, ")")
    
    result$min_count <- min_count
    result$concatenator <- meta(x, field = "concatenator", type = "object")
    result$call <- try(match.call(sys.function(-1), call = sys.call(-1)), silent = TRUE)
    result$version <- utils::packageVersion("wordvector")
    return(result)
}

word2vec <- function(...) {
    .Deprecated("textmodel_word2vec")
    textmodel_word2vec(...)
}

#' Print method for trained word vectors
#' @param x for print method, the object to be printed
#' @param ... unused
#' @method print textmodel_wordvector
#' @keywords internal
#' @return an invisible copy of `x`. 
#' @export
print.textmodel_wordvector <- function(x, ...) {
    cat("\nCall:\n")
    print(x$call)
    cat("\n", prettyNum(x$dim, big.mark = ","), " dimensions; ",
        prettyNum(nrow(x$values), big.mark = ","), " words.",
        "\n", sep = "")
    invisible(x)
}

#' Print method for trained document vectors
#' @param x for print method, the object to be printed
#' @param ... unused
#' @method print textmodel_docvector
#' @keywords internal
#' @return an invisible copy of `x`. 
#' @export
print.textmodel_docvector <- function(x, ...) {
    cat("\nCall:\n")
    print(x$call)
    cat("\n", prettyNum(x$dim, big.mark = ","), " dimensions; ",
        prettyNum(nrow(x$values), big.mark = ","), " documents.",
        "\n", sep = "")
    invisible(x)
}


#' Extract word vectors
#'
#' Extract word vectors from a `textmodel_wordvector` or `textmodel_docvector` object.
#' @param x a `textmodel_wordvector` or `textmodel_docvector` object.
#' @param ... not used
#' @return a matrix that contain the word vectors in rows
#' @export
as.matrix.textmodel_wordvector <- function(x, ...){
    return(x$values) 
}
