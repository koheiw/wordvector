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
                         std::function<void(float, float)> _progressCallback): m_threads() {
        trainThread_t::data_t data;

        if (!_settings) {
            throw std::runtime_error("train settings are not initialized");
        }
        data.settings = _settings;

        if (!_corpus) {
            throw std::runtime_error("corpus is object is not initialized");
        }
        data.corpus = _corpus;
        
        data.bpWeights.reset(new std::vector<float>(_settings->size * _corpus->types.size(), 0.0f));
        data.expTable.reset(new std::vector<float>(_settings->expTableSize));
        for (uint16_t i = 0; i < _settings->expTableSize; ++i) {
            // pre-compute the exp() table
            (*data.expTable)[i] = exp((i / static_cast<float>(_settings->expTableSize) * 2.0f - 1.0f) * _settings->expValueMax);
            // pre-compute f(x) = x / (x + 1)
            (*data.expTable)[i] = (*data.expTable)[i] / ((*data.expTable)[i] + 1.0f);
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
        m_matrixSize = data.settings->size * data.corpus->types.size();
        m_random = data.settings->random;
        
        for (uint8_t i = 0; i < _settings->threads; ++i) {
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
        
        for (auto &i:m_threads) {
            i->launch(_trainMatrix);
        }
        
        for (auto &i:m_threads) {
            i->join();
        }
    }
}
