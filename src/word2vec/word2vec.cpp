/**
 * @file
 * @brief
 * @author Max Fomichev
 * @date 15.02.2017
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/
#include <Rcpp.h>
#include "word2vec.hpp"
//#include "trainer.hpp"
#include "trainThread.hpp"

namespace w2v {
    bool word2vec_t::train(const settings_t &_settings,
                           const corpus_t &_corpus) noexcept {
        try {
            
            std::shared_ptr<corpus_t> corpus(new corpus_t(_corpus));
            std::shared_ptr<settings_t> settings(new settings_t(_settings));
            
            m_vectorSize = settings->size;
            m_vocaburarySize = corpus->words.size();
            
            // NOTE: to be replaced ---------------------------------------
            // train model
            //std::vector<float> _trainMatrix;
            // trainer_t(std::make_shared<settings_t>(_settings),
            //           corpus)(m_trainMatrix);
            
            // trainer_t::trainer_t() ---------------------------------------
            
            trainThread_t::data_t data;
            
            // if (!_settings) {
            //     throw std::runtime_error("train settings are not initialized");
            // }
            data.settings = settings;
            
            // if (!_corpus) {
            //     throw std::runtime_error("corpus is object is not initialized");
            // }
            data.corpus = corpus;
            
            
            data.bpWeights.reset(new std::vector<float>(settings->size * corpus->words.size(), 0.0f));
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
            
            // NOTE: consider setting size elsewhere
            std::size_t matrixSize = data.settings->size * data.corpus->words.size();
            uint32_t random = data.settings->random;
            int iter_max = data.settings->iterations;
            bool verbose = data.settings->verbose;
            
            std::vector<std::unique_ptr<trainThread_t>> threads;
            std::pair<std::size_t, std::size_t> range;
            std::size_t n = data.corpus->texts.size();
            for (uint16_t i = 0; i < settings->threads; ++i) {
                range = std::make_pair(floor((n / data.settings->threads) * i),
                                       floor((n / data.settings->threads) * (i + 1)) - 1);
                threads.emplace_back(new trainThread_t(range, data));
            }
            
            // trainer_t::operator() ---------------------------------------
            
            std::mt19937_64 randomGenerator(random);
            std::uniform_real_distribution<float> rndMatrixInitializer(-0.005f, 0.005f);
            m_trainMatrix.resize(matrixSize);
            // TODO: add m_bpWeights
            std::generate(m_trainMatrix.begin(), m_trainMatrix.end(), [&]() {
                return rndMatrixInitializer(randomGenerator);
            });
            int iter = 0;
            float alpha = 0.0;
            
            for (auto &thread:threads) {
                thread->launch(m_trainMatrix, iter, alpha);
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
            
            return true;
        } catch (const std::exception &_e) {
            m_errMsg = _e.what();
        } catch (...) {
            m_errMsg = "unknown error";
        }

        return false;
    }
}
