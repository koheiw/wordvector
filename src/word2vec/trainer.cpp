#include <Rcpp.h>
/**
 * @file
 * @brief trainer class of word2vec model
 * @author Max Fomichev
 * @date 20.12.2016
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/

#include "trainer.hpp"

namespace w2v {
    trainer_t::trainer_t(const std::shared_ptr<settings_t> &_settings,
                         const std::shared_ptr<corpus_t> &_corpus,
                         std::function<void(int, float)> _progressCallback): m_threads() {
        trainThread_t::data_t data;

        if (!_settings) {
            throw std::runtime_error("train settings are not initialized");
        }
        data.settings = _settings;

        if (!_corpus) {
            throw std::runtime_error("corpus is object is not initialized");
        }
        data.corpus = _corpus;
        
        data.bpWeights.reset(new std::vector<float>(_settings->size * _corpus->words.size(), 0.0f));
        data.expTable.reset(new std::vector<float>(_settings->expTableSize));
        for (uint16_t r = 0; r < _settings->expTableSize; ++r) {
            // scale value between +- expValueMax
            float s = exp((r / static_cast<float>(_settings->expTableSize) * 2.0f - 1.0f) * _settings->expValueMax);
            // pre-compute sigmoid: f(x) = exp(x) / (exp(x) + 1)
            (*data.expTable)[r] = s / (s + 1.0f);
            //std::cout << p << "," << s << "," << (*data.expTable)[p] << "\n";
        }
        
        if (_settings->withHS) {
            data.huffmanTree.reset(new huffmanTree_t(_corpus->frequency));;
        }

        if (_progressCallback != nullptr) {
            data.progressCallback = _progressCallback;
        }

        data.processedWords.reset(new std::atomic<std::size_t>(0));
        data.alpha.reset(new std::atomic<float>(_settings->alpha));
        
        // NOTE: consider setting size elsewhere
        m_matrixSize = data.settings->size * data.corpus->words.size();
        m_random = data.settings->random;
        m_iter = data.settings->iterations;
        
        for (uint16_t i = 0; i < _settings->threads; ++i) {
            m_threads.emplace_back(new trainThread_t(i, data));
        }
    }

    void trainer_t::operator()(std::vector<float> &_trainMatrix) noexcept {
        // input matrix initialized with small random values
        std::mt19937_64 randomGenerator(m_random);
        std::uniform_real_distribution<float> rndMatrixInitializer(-0.005f, 0.005f);
        _trainMatrix.resize(m_matrixSize);
        std::generate(_trainMatrix.begin(), _trainMatrix.end(), [&]() {
            return rndMatrixInitializer(randomGenerator);
        });
        int iter = 0;
        for (auto &i:m_threads) {
            i->launch(_trainMatrix, iter);
        }
        Rcpp::Rcout << "here1:" << iter << "\n";
        int iter_prev = 0;
        while (iter < m_iter) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (iter_prev < iter) {
                Rcpp::Rcout << "here2:" << iter << "\n"; // NOTE: use call back here?
                iter_prev = iter;
            }
        }
        for (auto &i:m_threads) {
            i->join();
        }
        Rcpp::Rcout << "here3:" << iter << "\n";
    }
}
