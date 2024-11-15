#include <Rcpp.h>
//#include <iostream>
//#include <iomanip>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <mutex>
#include "word2vec/word2vec.hpp"
#include "tokens.h"
//using namespace quanteda;

Rcpp::CharacterVector encode(std::vector<std::string> types){
    Rcpp::CharacterVector types_(types.size());
    for (std::size_t i = 0; i < types.size(); i++) {
        Rcpp::String type_ = types[i];
        type_.set_encoding(CE_UTF8);
        types_[i] = type_;
    }
    return(types_);
}

Rcpp::NumericMatrix as_matrix(w2v::w2vModel_t model) {
    
    std::unordered_map<std::string, std::vector<float>> m_map = model.map();
    std::vector<std::string> words;
    words.reserve(m_map.size());
    for(auto it : m_map) {
        words.push_back(it.first);
    } 

    std::vector<float> mat;
    mat.reserve(model.vectorSize() * words.size());
    for (size_t j = 0; j < words.size(); j++) {
        //auto p = model.vector(words[j]);
        auto it = m_map.find(words[j]);
        if (it != m_map.end()) {
            //std::vector<float> vec = *p;
            std::vector<float> vec = it->second;
            mat.insert(mat.end(), vec.begin(), vec.end());
        }
    }
    
    Rcpp::NumericMatrix mat_(model.vectorSize(), words.size(), mat.begin());
    colnames(mat_) = encode(words); 
    return Rcpp::transpose(mat_);
}

/*
 uint16_t minWordFreq = 5; ///< discard words that appear less than minWordFreq times
 uint16_t size = 100; ///< word vector size
 uint8_t window = 5; ///< skip length between words
 uint16_t expTableSize = 1000; ///< exp(x) / (exp(x) + 1) values lookup table size
 uint8_t expValueMax = 6; ///< max value in the lookup table
 float sample = 1e-3f; ///< threshold for occurrence of words
 bool withHS = false; ///< use hierarchical softmax instead of negative sampling
 uint8_t negative = 5; ///< negative examples number
 uint8_t threads = 12; ///< train threads number
 uint8_t iterations = 5; ///< train iterations
 float alpha = 0.05f; ///< starting learn rate
 bool withSG = false; ///< use Skip-Gram instead of CBOW
*/

// [[Rcpp::export]]
Rcpp::List cpp_w2v(Rcpp::List texts_, 
                   Rcpp::CharacterVector types_, 
                   uint16_t size = 100,
                   uint8_t window = 5,
                   uint16_t expTableSize = 1000,
                   uint8_t expValueMax = 6,
                   float sample = 0.001,
                   bool withHS = false,
                   uint8_t negative = 5,
                   uint8_t threads = 1,
                   uint8_t iterations = 5,
                   float alpha = 0.05,
                   bool withSG = false,
                   bool verbose = false,
                   bool normalize = true) {
  
    if (verbose) {
        if (withSG) {
            Rprintf("Training Skip-gram model with %d dimensions\n", size);
        } else {
            Rprintf("Training CBOW model with %d dimensions\n", size);
        }
        Rprintf(" ...using %d threads for distributed computing\n", threads);
        Rprintf(" ...initializing\n");
    }

    texts_t texts = Rcpp::as<texts_t>(texts_);
    types_t types = Rcpp::as<types_t>(types_);
    //texts_t texts = xptr->texts;
    //types_t types = xptr->types;
    
    w2v::corpus_t corpus(texts, types);
    corpus.setWordFreq();
      
    w2v::trainSettings_t ts;
    ts.size = size;
    ts.window = window;
    ts.expTableSize = expTableSize;
    ts.expValueMax = expValueMax;
    ts.sample = sample;
    ts.withHS = withHS;
    ts.negative = negative;
    ts.threads = threads > 0 ? threads : std::thread::hardware_concurrency();
    ts.iterations = iterations;
    ts.alpha = alpha;
    ts.withSG = withSG;
    ts.random = (uint32_t)(Rcpp::runif(1)[0] * std::numeric_limits<uint32_t>::max());

    w2v::w2vModel_t model;
    bool trained;
  
    if (verbose) {
        if (withHS) {
            Rprintf(" ...Hierarchical Softmax in %d iterations\n", iterations);
        } else {
            Rprintf(" ...Negative Sampling in %d iterations\n", iterations);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        int iter = 0;
        std::mutex mtx;
        trained = model.train(ts, corpus, [&start, &iter, &mtx] (int _iter, float _alpha) {
        mtx.lock();
        if (_iter > iter) {
            iter = _iter;
            auto end = std::chrono::high_resolution_clock::now();
            auto diff = std::chrono::duration<double, std::milli>(end - start);
            double msec = diff.count();
            Rprintf(" ......iteration %d ", iter);
            Rprintf("elapsed time: %.2f seconds (alpha: %.4f)\n", msec / 1000, _alpha);
        };
        mtx.unlock();
        });
    } else {
        trained = model.train(ts, corpus, nullptr);
    }
    
    if (!trained) {
        Rcpp::List out = Rcpp::List::create(
            Rcpp::Named("message") = model.errMsg()
        );
        return out;
    }
    // NORMALISE UPFRONT - DIFFERENT THAN ORIGINAL CODE 
    // - original code dumps data to disk, next imports it and during import normalisation happens after which we can do nearest calculations
    // - the R wrapper only writes to disk at request so we need to normalise upfront in order to do directly nearest calculations
    if (normalize) {
        model.normalize();
        if (verbose)
            Rprintf(" ...normalizing vectors\n");
    }
    if (verbose)
        Rprintf(" ...complete\n");
    
    // Return model + model information
    Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model") = as_matrix(model),
    //Rcpp::Named("model") = model,
    //Rcpp::Named("vocabulary") = types.size(),
    //Rcpp::Named("success") = success,
    //Rcpp::Named("error_log") = model.errMsg(),
    Rcpp::Named("dim") = size,
    Rcpp::Named("type") = "", // placeholder 
    Rcpp::Named("min_count") = 0L, // placeholder 
    Rcpp::Named("window") = window,
    Rcpp::Named("iter") = iterations,
    Rcpp::Named("lr") = alpha,
    Rcpp::Named("hs") = withHS,
    Rcpp::Named("negative") = negative,
    Rcpp::Named("sample") = sample
    // NOTE: move to R
    // Rcpp::Named("control") = Rcpp::List::create(
    //     //Rcpp::Named("dim") = size,
    //     //Rcpp::Named("window") = window,
    //     //Rcpp::Named("iter") = iterations,
    //     //Rcpp::Named("lr") = alpha,
    //     //Rcpp::Named("skipgram") = withSG,
    //     //Rcpp::Named("hs") = withHS,
    //     //Rcpp::Named("negative") = negative,
    //     //Rcpp::Named("sample") = sample,
    //     //Rcpp::Named("expTableSize") = expTableSize,
    //     //Rcpp::Named("expValueMax") = expValueMax
    // )
    );
    out.attr("class") = "textmodel_wordvector";
    return out;
}

