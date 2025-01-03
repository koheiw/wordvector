#include <Rcpp.h>
//#include <iostream>
//#include <iomanip>
#include <chrono>
#include <thread>
//#include <unordered_map>
#include <mutex>
#include "word2vec/word2vec.hpp"
#include "tokens.h"

Rcpp::CharacterVector encode(std::vector<std::string> types){
    Rcpp::CharacterVector types_(types.size());
    for (std::size_t i = 0; i < types.size(); i++) {
        Rcpp::String type_ = types[i];
        type_.set_encoding(CE_UTF8);
        types_[i] = type_;
    }
    return(types_);
}

Rcpp::NumericMatrix get_values(w2v::word2vec_t model, w2v::corpus_t corpus) {
    std::vector<float> mat = model.values();
    if (model.vectorSize() * model.vocaburarySize() != mat.size())
        throw std::runtime_error("Invalid model values");
    Rcpp::NumericMatrix mat_(model.vectorSize(), model.vocaburarySize(), mat.begin());
    colnames(mat_) = encode(corpus.words); 
    return Rcpp::transpose(mat_);
}

Rcpp::NumericMatrix get_weights(w2v::word2vec_t model, w2v::corpus_t corpus) {
    std::vector<float> mat = model.weights();
    if (model.vectorSize() * model.vocaburarySize() != mat.size())
        throw std::runtime_error("Invalid model weights");
    Rcpp::NumericMatrix mat_(model.vectorSize(), model.vocaburarySize(), mat.begin());
    colnames(mat_) = encode(corpus.words); 
    return Rcpp::transpose(mat_);
}

Rcpp::NumericVector get_frequency(w2v::corpus_t corpus) {
    Rcpp::NumericVector v = Rcpp::wrap(corpus.frequency);
    v.names() = encode(corpus.words);
    return(v);
}

/*
 uint16_t minWordFreq = 5; ///< discard words that appear less than minWordFreq times
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
Rcpp::List cpp_w2v(Rcpp::List texts_, 
                   Rcpp::CharacterVector words_, 
                   uint16_t minWordFreq = 5,
                   uint16_t size = 100,
                   uint16_t window = 5,
                   float sample = 0.001,
                   bool withHS = false,
                   uint16_t negative = 5,
                   uint16_t threads = 1,
                   uint16_t iterations = 5,
                   float alpha = 0.05,
                   int type = 1,
                   bool verbose = false,
                   bool normalize = true,
                   uint16_t expTableSize = 1000,
                   uint16_t expValueMax = 6) {
  
    if (verbose) {
        if (type == 1 || type == 10) {
            Rprintf("Training CBOW model with %d dimensions\n", size);
        } else if (type == 2 || type == 20) {
            Rprintf("Training skip-gram model with %d dimensions\n", size);
        }
        Rprintf(" ...using %d threads for distributed computing\n", threads);
        Rprintf(" ...initializing\n");
    }

    texts_t texts = Rcpp::as<texts_t>(texts_);
    words_t words = Rcpp::as<words_t>(words_);
    //texts_t texts = xptr->texts;
    //types_t types = xptr->types;
    
    w2v::corpus_t corpus(texts, words);
    corpus.setWordFreq();
      
    w2v::settings_t settings;
    settings.minWordFreq = minWordFreq;
    settings.size = size;
    settings.window = window;
    settings.expTableSize = expTableSize;
    settings.expValueMax = expValueMax;
    settings.sample = sample;
    settings.withHS = withHS;
    settings.negative = negative;
    settings.threads = threads > 0 ? threads : std::thread::hardware_concurrency();
    settings.iterations = iterations;
    settings.alpha = alpha;
    settings.type = type;
    settings.random = (uint32_t)(Rcpp::runif(1)[0] * std::numeric_limits<uint32_t>::max());
    settings.verbose = verbose;

    w2v::word2vec_t word2vec;
    bool trained;
    
    if (verbose) {
        if (withHS) {
            Rprintf(" ...hierarchical softmax in %d iterations\n", iterations);
        } else {
            Rprintf(" ...negative sampling in %d iterations\n", iterations);
        }
    }    
    trained = word2vec.train(settings, corpus);
    
    if (!trained) {
        Rcpp::List out = Rcpp::List::create(
            Rcpp::Named("message") = word2vec.errMsg()
        );
        return out;
    }
    if (normalize) {
        word2vec.normalize();
        if (verbose)
            Rprintf(" ...normalizing vectors\n");
    }
    if (verbose)
        Rprintf(" ...complete\n");
    
    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("values") = get_values(word2vec, corpus), 
        Rcpp::Named("weights") = get_weights(word2vec, corpus), 
        Rcpp::Named("type") = type,
        Rcpp::Named("dim") = size,
        Rcpp::Named("min_count") = minWordFreq,
        Rcpp::Named("frequency") = get_frequency(corpus),
        Rcpp::Named("window") = window,
        Rcpp::Named("iter") = iterations,
        Rcpp::Named("alpha") = alpha,
        Rcpp::Named("use_ns") = !withHS,
        Rcpp::Named("ns_size") = negative,
        Rcpp::Named("sample") = sample
    );
    out.attr("class") = "textmodel_wordvector";
    return out;
}

