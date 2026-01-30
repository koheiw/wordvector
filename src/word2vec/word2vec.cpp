/**
 * @file
 * @brief
 * @author Max Fomichev
 * @date 15.02.2017
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/
#include <Rcpp.h>
#include "word2vec.hpp"
#include "trainThread.hpp"

namespace w2v {
    bool word2vec_t::train(const settings_t &_settings,
                           const corpus_t &_corpus,
                           const word2vec_t &_model) noexcept {
        try {
            
            std::shared_ptr<corpus_t> corpus(new corpus_t(_corpus));
            std::shared_ptr<settings_t> settings(new settings_t(_settings));
            
            m_vocabulary = corpus->types;
            m_vocabularySize = corpus->types.size();
            m_vectorSize = settings->size;
            m_corpusSize = corpus->texts.size();
            
            std::size_t matrixSize = m_vectorSize * m_vocabularySize;
            std::size_t docMatrixSize = 0;
            if (settings->type > 2) 
                docMatrixSize = m_vectorSize * m_corpusSize;
            std::mt19937_64 randomGenerator(settings->random);
            int iter_max = settings->iterations;
            bool verbose = settings->verbose;
            
            if (m_vectorSize == 0)
                throw std::runtime_error("vectorSize is zero");
                
            if (m_vocabularySize == 0)
                throw std::runtime_error("vocabularySize is zero");
            
            if (_corpus.trainWords == 0)
                throw std::runtime_error("trainWords is zero");
            
            // set data
            trainThread_t::data_t data;
            data.settings = settings;
            data.corpus = corpus;
            
            // initialize variables
            double rndMin, rndMax;
            if (settings->initMin) {
                rndMin = settings->expValueMax * -1;
                rndMax = settings->expValueMax * -1 + 0.01f;
            } else {
                rndMin = -0.005f;
                rndMax =  0.005f;
            }
            std::uniform_real_distribution<float> rndMatrixInitializer(rndMin, rndMax);
            
            // word vector
            data.bpWeights.reset(new std::vector<float>(matrixSize, 0.0f));
            data.pjLayerValues.reset(new std::vector<float>(matrixSize, 0.0f));
            std::generate((*data.pjLayerValues).begin(), (*data.pjLayerValues).end(), [&]() {
                return rndMatrixInitializer(randomGenerator);
            });
            // document vector
            data.docValues.reset(new std::vector<float>(docMatrixSize, 0.0f));
            std::generate((*data.docValues).begin(), (*data.docValues).end(), [&]() {
                return rndMatrixInitializer(randomGenerator);
            });
            data.expTable.reset(new std::vector<float>(settings->expTableSize));
            for (uint16_t r = 0; r < settings->expTableSize; ++r) {
                // scale value between +- expValueMax
                float s = exp((r / static_cast<float>(settings->expTableSize) * 2.0f - 1.0f) * settings->expValueMax);
                // pre-compute sigmoid: f(x) = exp(x) / (exp(x) + 1)
                (*data.expTable)[r] = s / (s + 1.0f);
                //std::cout << p << "," << s << "," << (*data.expTable)[p] << "\n";
            }
            
            if (settings->withHS) {
                data.huffmanTree.reset(new huffmanTree_t(corpus->frequency));;
            }
            data.processedWords.reset(new std::atomic<std::size_t>(0));
            data.alpha.reset(new std::atomic<float>(settings->alpha));
            
            // inherit parameters
            if (_model.m_vocabulary.size() > 0) {
                if (verbose) {
                    Rprintf(" ......copy pre-trained word vectors\n");
                }
                std::unordered_map<std::string, std::size_t> map;
                for (std::size_t i = 0; i < m_vocabularySize; ++i) {
                    map.insert(std::make_pair(m_vocabulary[i], i));
                }
                for (std::size_t j = 0; j < _model.m_vocabularySize; ++j) {
                    if (auto it = map.find(_model.m_vocabulary[j]); it != map.end()) {
                        for (std::size_t k = 0; k < m_vectorSize; k++) {
                            (*data.pjLayerValues)[k + (it->second * m_vectorSize)] = _model.m_pjLayerValues[k + (j * _model.m_vectorSize)];
                            (*data.bpWeights)[k + (it->second * m_vectorSize)] = _model.m_bpWeights[k + (j * _model.m_vectorSize)];
                        }
                    }
                }
            }
            
            // create threads
            std::vector<std::unique_ptr<trainThread_t>> threads;
            std::size_t n = data.corpus->texts.size();
            std::size_t per = ceil((float)n / (float)data.settings->threads);
            for (std::size_t i = 0; i < settings->threads; ++i) {
                std::size_t from = per * i;
                std::size_t to = std::min(per * (i + 1) - 1, n - 1);
                //Rcpp::Rcout << settings->threads << " " << per << " " << from << " " << to << "\n";
                threads.emplace_back(new trainThread_t(std::make_pair(from, to), data));
                if (n - 1 == to) 
                    break;
            }
            
            if (verbose) {
                if (settings->withHS) {
                    Rprintf(" ...hierarchical softmax in %d iterations\n", 
                            settings->iterations);
                } else {
                    Rprintf(" ...negative sampling in %d iterations\n", 
                            settings->iterations);
                }
            } 
            
            int iter = 0;
            float alpha = 0.0;
            for (auto &thread:threads) {
                thread->launch(iter, alpha);
            }
            
            if (verbose) {
                int iter_prev = 0;
                auto start = std::chrono::high_resolution_clock::now();
                while (iter < iter_max) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    if (iter_prev < iter) {
                        auto end = std::chrono::high_resolution_clock::now();
                        auto diff = std::chrono::duration<double, std::milli>(end - start);
                        double msec = diff.count();
                        Rprintf(" ......iteration %d elapsed time: %.2f seconds (alpha: %.4f)\n",
                                iter, msec / 1000, alpha);
                        iter_prev = iter;
                    }
                }
            }
            
            for (auto &thread:threads) {
                thread->join();
            }
            
            m_pjLayerValues = *data.pjLayerValues;
            m_bpWeights = *data.bpWeights;
            m_docValues = *data.docValues;
            
            return true;
            
        } catch (const std::exception &_e) {
            m_errMsg = _e.what();
        } catch (...) {
            m_errMsg = "unknown error";
        }

        return false;
    }
}
