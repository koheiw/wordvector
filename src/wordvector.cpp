#include <Rcpp.h>
#include <chrono>
#include <thread>
#include <mutex>
#include "word2vec/word2vec.hpp"
#include "tokens.h"

typedef XPtr<TokensObj> TokensPtr;
typedef std::vector<std::string> vocabulary_t;
typedef std::vector<float> wordvector_t;


Rcpp::CharacterVector encode(std::vector<std::string> types){
    Rcpp::CharacterVector types_(types.size());
    for (std::size_t i = 0; i < types.size(); i++) {
        Rcpp::String type_ = types[i];
        type_.set_encoding(CE_UTF8);
        types_[i] = type_;
    }
    return types_;
}

Rcpp::NumericMatrix get_words(w2v::word2vec_t model) {
    std::vector<float> mat = model.values();
    if (model.vectorSize() * model.vocabularySize() != mat.size())
        throw std::runtime_error("Invalid word matrix");
    Rcpp::NumericMatrix mat_(model.vectorSize(), model.vocabularySize(), mat.begin());
    colnames(mat_) = encode(model.vocabulary()); 
    return Rcpp::transpose(mat_);
}

Rcpp::NumericMatrix get_documents(w2v::word2vec_t model) {
    std::vector<float> mat = model.docValues();
    if (mat.size() == 0)
        return Rcpp::NumericMatrix();
    if (model.vectorSize() * model.corpusSize() != mat.size())
        throw std::runtime_error("Invalid document matrix");
    Rcpp::NumericMatrix mat_(model.vectorSize(), model.corpusSize(), mat.begin());
    return Rcpp::transpose(mat_);
}

Rcpp::NumericMatrix get_weights(w2v::word2vec_t model) {
    std::vector<float> mat = model.weights();
    if (model.vectorSize() * model.vocabularySize() != mat.size())
        throw std::runtime_error("Invalid weight matrix");
    Rcpp::NumericMatrix mat_(model.vectorSize(), model.vocabularySize(), mat.begin());
    colnames(mat_) = encode(model.vocabulary()); 
    return Rcpp::transpose(mat_);
}

Rcpp::NumericVector get_frequency(w2v::corpus_t corpus) {
    Rcpp::NumericVector vec_ = Rcpp::wrap(corpus.frequency);
    vec_.names() = encode(corpus.types);
    return vec_;
}

w2v::word2vec_t as_word2vec(List model_) {
    
    w2v::word2vec_t model;
    if (model_.length() == 0)
        return model;
    
    std::size_t dim;
    vocabulary_t vocabulary;
    wordvector_t words, weights;
    
    // vector sizes
    dim = as<Rcpp::IntegerVector>(model_["dim"])[0];
    
    // vocabulary
    CharacterVector vocabulary_ = as<Rcpp::NumericVector>(model_["frequency"]).names();
    vocabulary = Rcpp::as<vocabulary_t>(vocabulary_);
    
    // word vectors
    Rcpp::List values_ = model_["values"];
    if (values_.containsElementNamed("word")) {
        Rcpp::NumericMatrix words_ = values_["word"];
        words_ = Rcpp::transpose(words_); // columns are words internally
        words = Rcpp::as<wordvector_t>(NumericVector(words_));
    } else {
        words = wordvector_t(dim * vocabulary.size()); // TODO: change to empty matrix
    }
    
    // weights
    Rcpp::NumericMatrix weights_ = model_["weights"];
    weights_ = Rcpp::transpose(weights_); // columns are words internally
    weights = Rcpp::as<wordvector_t>(NumericVector(weights_));
    
    model = w2v::word2vec_t(vocabulary, dim, words, weights);
    return model;
}

/*
 uint16_t size = 100; ///< word vector size
 uint16_t window = 5; ///< skip length between words
 uint16_t expTableSize = 1000; ///< exp(x) / (exp(x) + 1) values lookup table size
 uint16_t expValueMax = 6; ///< max value in the lookup table
 float sample = 1e-3f; ///< threshold for occurrence of words
 bool withHS = false; ///< use hierarchical softmax instead of negative sampling
 uint16_t negative = 5; ///< negative examples number
 uint16_t threads = 1; ///< train threads number
 uint16_t iterations = 5; ///< train iterations
 float alpha = 0.05f; ///< starting learn rate
 int type = 1; ///< 1:CBOW 2:Skip-Gram
*/

// [[Rcpp::export]]
Rcpp::List cpp_word2vec(TokensPtr xptr, 
                        List model,
                        uint16_t size = 100,
                        uint16_t window = 5,
                        float sample = 0.001,
                        bool withHS = false,
                        uint16_t negative = 5,
                        uint16_t threads = 1,
                        uint16_t iterations = 5,
                        float alpha = 0.05,
                        int type = 1,
                        bool doc2vec = false,
                        bool verbose = false,
                        bool normalize = true) {
  
    if (verbose) {
        if (type == 1) {
            Rprintf("Training continuous BOW model with %d dimensions\n", size);
        } else if (type == 2) {
            Rprintf("Training skip-gram model with %d dimensions\n", size);
        } else if (type == 3) {
            Rprintf("Training distributed memory model with %d dimensions\n", size);
        } else if (type == 4) {
            Rprintf("Training distributed BOW model with %d dimensions\n", size);
        } else {
            Rprintf("Training type %d model with %d dimensions\n", type, size);
        }
        Rprintf(" ...using %d threads for distributed computing\n", threads);
        Rprintf(" ...initializing\n");
    }
    
    xptr->recompile();
    texts_t texts = xptr->texts;
    types_t types = xptr->types;
    
    w2v::corpus_t corpus(texts, types);
    corpus.setWordFreq();
      
    w2v::settings_t settings;;
    settings.size = size;
    settings.window = window;
    settings.expTableSize = 1000;
    settings.expValueMax = 6;
    settings.sample = sample;
    settings.withHS = withHS;
    settings.negative = negative;
    settings.threads = threads > 0 ? threads : std::thread::hardware_concurrency();
    settings.iterations = iterations;
    settings.alpha = alpha;
    settings.type = type;
    settings.random = (uint32_t)(Rcpp::runif(1)[0] * std::numeric_limits<uint32_t>::max());
    settings.verbose = verbose;
    
    // NOTE: consider initializing models with corpus
    w2v::word2vec_t word2vec_pre = as_word2vec(model);
    w2v::word2vec_t word2vec;
    bool trained;
    
    trained = word2vec.train(settings, corpus, word2vec_pre);
    
    if (!trained) {
        Rcpp::List out = Rcpp::List::create(
            Rcpp::Named("message") = word2vec.errMsg()
        );
        return out;
    }
    if (verbose)
        Rprintf(" ...complete\n");
    
    Rcpp::List values;
    if (type == 3) { // dm
        values = Rcpp::List::create(
            Rcpp::Named("word") = get_words(word2vec), 
            Rcpp::Named("doc") = get_documents(word2vec)
        );
    } else if (type == 4) { // dbow
        values = Rcpp::List::create(
            Rcpp::Named("doc") = get_documents(word2vec)
        );
    } else { // cbow or dbow
        values = Rcpp::List::create(
            Rcpp::Named("word") = get_words(word2vec)
        );
    }
    Rcpp::List res = Rcpp::List::create(
        Rcpp::Named("values") = values,
        Rcpp::Named("weights") = get_weights(word2vec), 
        Rcpp::Named("type") = type,
        Rcpp::Named("dim") = size,
        Rcpp::Named("frequency") = get_frequency(corpus),
        Rcpp::Named("window") = window,
        Rcpp::Named("iter") = iterations,
        Rcpp::Named("alpha") = alpha,
        Rcpp::Named("use_ns") = !withHS,
        Rcpp::Named("ns_size") = negative,
        Rcpp::Named("sample") = sample,
        Rcpp::Named("normalize") = normalize
    );
    return res;
}
