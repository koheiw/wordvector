#' @title Train a word2vec model on text
#' @description Construct a word2vec model on text. The algorithm is explained at \url{https://arxiv.org/pdf/1310.4546.pdf}
#' @param x a character vector with text or the path to the file on disk containing training data or a list of tokens. See the examples.
#' @param type the type of algorithm to use, either 'cbow' or 'skip-gram'. Defaults to 'cbow'
#' @param dim dimension of the word vectors. Defaults to 50.
#' @param iter number of training iterations. Defaults to 5.
#' @param lr initial learning rate also known as alpha. Defaults to 0.05
#' @param window skip length between words. Defaults to 5.
#' @param hs logical indicating to use hierarchical softmax instead of negative sampling. Defaults to FALSE indicating to do negative sampling.
#' @param negative integer with the number of negative samples. Only used in case hs is set to FALSE
#' @param sample threshold for occurrence of words. Defaults to 0.001
#' @param min_count integer indicating the number of time a word should occur to be considered as part of the training vocabulary. Defaults to 5.
#' @param stopwords a character vector of stopwords to exclude from training 
#' @param threads number of CPU threads to use. Defaults to 1.
#' @param ... further arguments passed on to the methods \code{\link{word2vec.character}}, \code{\link{word2vec.list}} as well as the C++ function \code{w2v_train} - for expert use only
#' @return an object of class \code{w2v_trained} which is a list with elements 
#' \itemize{
#' \item{model: a Rcpp pointer to the model}
#' \item{data: a list with elements file: the training data used, stopwords: the character vector of stopwords, n}
#' \item{vocabulary: the number of words in the vocabulary}
#' \item{success: logical indicating if training succeeded}
#' \item{error_log: the error log in case training failed}
#' \item{control: as list of the training arguments used, namely min_count, dim, window, iter, lr, skipgram, hs, negative, sample, split_words, split_sents, expTableSize and expValueMax}
#' }
#' @references \url{https://github.com/maxoodf/word2vec}, \url{https://arxiv.org/pdf/1310.4546.pdf}
#' @details 
#' Some advice on the optimal set of parameters to use for training as defined by Mikolov et al.
#' \itemize{
#' \item{argument type: skip-gram (slower, better for infrequent words) vs cbow (fast)}
#' \item{argument hs: the training algorithm: hierarchical softmax (better for infrequent words) vs negative sampling (better for frequent words, better with low dimensional vectors)}
#' \item{argument dim: dimensionality of the word vectors: usually more is better, but not always}
#' \item{argument window: for skip-gram usually around 10, for cbow around 5}
#' \item{argument sample: sub-sampling of frequent words: can improve both accuracy and speed for large data sets (useful values are in range 0.001 to 0.00001)}
#' }
#' @note
#' Some notes on the tokenisation
#' \itemize{
#' \item{If you provide to \code{x} a list, each list element should correspond to a sentence (or what you consider as a sentence) and should contain a character vector of tokens. The word2vec model is then executed using \code{\link{word2vec.list}}}
#' \item{If you provide to \code{x} a character vector or the path to the file on disk, the tokenisation into words depends on the first element provided in \code{split} and the tokenisation into sentences depends on the second element provided in \code{split} when passed on to \code{\link{word2vec.character}}}
#' }
#' @seealso \code{\link{predict.word2vec}}, \code{\link{as.matrix.word2vec}}, \code{\link{word2vec}}, \code{\link{word2vec.character}}, \code{\link{word2vec.list}}
#' @export
#' @useDynLib wordvector
#' @examples
#' \dontshow{if(require(udpipe))\{}
#' library(udpipe)
#' ## Take data and standardise it a bit
#' data(brussels_reviews, package = "udpipe")
#' x <- subset(brussels_reviews, language == "nl")
#' x <- tolower(x$feedback)
#' 
#' ## Build the model get word embeddings and nearest neighbours
#' model <- word2vec(x = x, dim = 15, iter = 20)
#' emb   <- as.matrix(model)
#' head(emb)
#' emb   <- predict(model, c("bus", "toilet", "unknownword"), type = "embedding")
#' emb
#' nn    <- predict(model, c("bus", "toilet"), type = "nearest", top_n = 5)
#' nn
#' 
#' ## Get vocabulary
#' vocab   <- summary(model, type = "vocabulary")
#' 
#' # Do some calculations with the vectors and find similar terms to these
#' emb     <- as.matrix(model)
#' vector  <- emb["buurt", ] - emb["rustige", ] + emb["restaurants", ]
#' predict(model, vector, type = "nearest", top_n = 10)
#' 
#' vector  <- emb["gastvrouw", ] - emb["gastvrij", ]
#' predict(model, vector, type = "nearest", top_n = 5)
#' 
#' vectors <- emb[c("gastheer", "gastvrouw"), ]
#' vectors <- rbind(vectors, avg = colMeans(vectors))
#' predict(model, vectors, type = "nearest", top_n = 10)
#' 
#' ## Save the model to hard disk
#' path <- "mymodel.bin"
#' \dontshow{
#' path <- tempfile(pattern = "w2v", fileext = ".bin")
#' }
#' write.word2vec(model, file = path)
#' model <- read.word2vec(path)
#' 
#' \dontshow{
#' file.remove(path)
#' }
#' ## 
#' ## Example of word2vec with a list of tokens 
#' ## 
#' toks  <- strsplit(x, split = "[[:space:][:punct:]]+")
#' model <- word2vec(x = toks, dim = 15, iter = 20)
#' emb   <- as.matrix(model)
#' emb   <- predict(model, c("bus", "toilet", "unknownword"), type = "embedding")
#' emb
#' nn    <- predict(model, c("bus", "toilet"), type = "nearest", top_n = 5)
#' nn
#' 
#' ## 
#' ## Example getting word embeddings 
#' ##   which are different depending on the parts of speech tag
#' ## Look to the help of the udpipe R package 
#' ##   to get parts of speech tags on text
#' ## 
#' library(udpipe)
#' data(brussels_reviews_anno, package = "udpipe")
#' x <- subset(brussels_reviews_anno, language == "fr")
#' x <- subset(x, grepl(xpos, pattern = paste(LETTERS, collapse = "|")))
#' x$text <- sprintf("%s/%s", x$lemma, x$xpos)
#' x <- subset(x, !is.na(lemma))
#' x <- split(x$text, list(x$doc_id, x$sentence_id))
#' 
#' model <- word2vec(x = x, dim = 15, iter = 20)
#' emb   <- as.matrix(model)
#' nn    <- predict(model, c("cuisine/NN", "rencontrer/VB"), type = "nearest")
#' nn
#' nn    <- predict(model, c("accueillir/VBN", "accueillir/VBG"), type = "nearest")
#' nn
#' 
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
word2vec <- function(x, 
                     type = c("cbow", "skip-gram"),
                     dim = 50, window = ifelse(type == "cbow", 5L, 10L), 
                     iter = 5L, lr = 0.05, hs = FALSE, negative = 5L, sample = 0.001, min_count = 5L, 
                     stopwords = character(),
                     threads = 1L,
                     ...) {
    UseMethod("word2vec")
}

#' @inherit word2vec title description params details seealso return references
#' @export
#' @importFrom quanteda dfm featfreq tokens_keep
#' @examples 
#' \dontshow{if(require(udpipe))\{}
#' library(udpipe)
#' data(brussels_reviews, package = "udpipe")
#' x     <- subset(brussels_reviews, language == "nl")
#' x     <- tolower(x$feedback)
#' toks  <- strsplit(x, split = "[[:space:][:punct:]]+")
#' model <- word2vec(x = toks, dim = 15, iter = 20)
#' emb   <- as.matrix(model)
#' head(emb)
#' emb   <- predict(model, c("bus", "toilet", "unknownword"), type = "embedding")
#' emb
#' nn    <- predict(model, c("bus", "toilet"), type = "nearest", top_n = 5)
#' nn
#' 
#' ## 
#' ## Example of word2vec with a list of tokens
#' ## which gives the same embeddings as with a similarly tokenised character vector of texts 
#' ## 
#' txt   <- txt_clean_word2vec(x, ascii = TRUE, alpha = TRUE, tolower = TRUE, trim = TRUE)
#' table(unlist(strsplit(txt, "")))
#' toks  <- strsplit(txt, split = " ")
#' set.seed(1234)
#' modela <- word2vec(x = toks, dim = 15, iter = 20)
#' set.seed(1234)
#' modelb <- word2vec(x = txt, dim = 15, iter = 20, split = c(" \n\r", "\n\r"))
#' all.equal(as.matrix(modela), as.matrix(modelb))
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
word2vec.tokens <- function(x,
                          type = c("cbow", "skip-gram"),
                          dim = 50, window = ifelse(type == "cbow", 5L, 10L), 
                          iter = 5L, lr = 0.05, hs = FALSE, negative = 5L, sample = 0.001, min_count = 5L, 
                          threads = 1L,
                          ...){
    
    type <- match.arg(type)
    #expTableSize <- 1000L
    #expValueMax <- 6L
    #expTableSize <- as.integer(expTableSize)
    #expValueMax <- as.integer(expValueMax)
    min_count <- as.integer(min_count)
    dim <- as.integer(dim)
    window <- as.integer(window)
    iter <- as.integer(iter)
    sample <- as.numeric(sample)
    hs <- as.logical(hs)
    negative <- as.integer(negative)
    threads <- as.integer(threads)
    iter <- as.integer(iter)
    lr <- as.numeric(lr)
    skipgram <- as.logical(type %in% "skip-gram")
    
    # NOTE: use tokens_xptr
    #x <- as.tokenx_xptr(x)
    x <- tokens_trim(x, min_termfreq = min_count, termfreq_type = "count")
    result <- cpp_w2v(x, attr(x, "types"), 
                     size = dim, window = window,
                     sample = sample, withHS = hs, negative = negative, 
                     threads = threads, iterations = iter,
                     alpha = lr, withSG = skipgram, ...)
    return(result)
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
as.matrix.textmodel_word2vec <- function(x, ...){
    return(x$model) 
}

#' @export
as.matrix.textmodel_doc2vec <- function(x, ...){
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
    if (is.null(model))
        model <- word2vec(x, ...)
    wov <- as.matrix(model)
    dfmt <- dfm(x)
    dfmt <- dfm_match(dfmt, rownames(wov))
    dov <- Matrix::tcrossprod(dfmt, t(wov)) # NOTE: consider using proxyC
    model$model <- dov / sqrt(Matrix::rowSums(dov ^ 2) / ncol(dov))
    class(model) <- "textmodel_doc2vec"
    return(model)
}

#' @examples
#' analogy(mod, ~ japan - tokyo)
#' @export
analogy <- function(model, formula, n = 10) {
    
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
    s <- Matrix::rowSums(proxyC::simil(emb, v, margin = 2, use_nan = TRUE))
    # without normalization (from word2vec)
    # suppressWarnings({
    #     s <- rowSums(sqrt(crossprod(emb, v) / nrow(emb))) 
    # })
    s <- head(sort(s, decreasing = TRUE), n)
    res <- data.frame(word = names(s), 
                      cosine = s, row.names = NULL)
    attr(res, "formula") <- formula
    attr(res, "weight") <- weight
    return(res)
}



#' @export
synonyms <- function(model, terms, n = 10) { # NOTE: consider changing to neighbors()
    emb <- as.matrix(model)
    terms <- intersect(terms, rownames(emb))
    sim <- proxyC::simil(emb[terms,,drop = FALSE], emb)
    sapply(terms, function(term) {
        head(names(sort(Matrix::colSums(sim[term,,drop = FALSE]), decreasing = TRUE)), n)
    })
}
