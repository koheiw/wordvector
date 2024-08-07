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
    trainer_t::trainer_t(const std::shared_ptr<trainSettings_t> &_trainSettings,
                         const std::shared_ptr<corpus_t> &_corpus,
                         std::function<void(float, float)> _progressCallback): m_threads() {
        trainThread_t::sharedData_t sharedData;

        if (!_trainSettings) {
            throw std::runtime_error("train settings are not initialized");
        }
        sharedData.trainSettings = _trainSettings;

        if (!_corpus) {
            throw std::runtime_error("corpus is object is not initialized");
        }
        sharedData.corpus = _corpus;
        
        sharedData.bpWeights.reset(new std::vector<float>(_trainSettings->size * _corpus->types.size(), 0.0f));
        sharedData.expTable.reset(new std::vector<float>(_trainSettings->expTableSize));
        for (uint16_t i = 0; i < _trainSettings->expTableSize; ++i) {
            // Precompute the exp() table
            (*sharedData.expTable)[i] =
                    exp((i / static_cast<float>(_trainSettings->expTableSize) * 2.0f - 1.0f)
                                           * _trainSettings->expValueMax);
            // Precompute f(x) = x / (x + 1)
            (*sharedData.expTable)[i] = (*sharedData.expTable)[i] / ((*sharedData.expTable)[i] + 1.0f);
        }
        
        if (_trainSettings->withHS) {
            sharedData.huffmanTree.reset(new huffmanTree_t(_corpus->frequency));;
        }

        if (_progressCallback != nullptr) {
            sharedData.progressCallback = _progressCallback;
        }

        sharedData.processedWords.reset(new std::atomic<std::size_t>(0));
        sharedData.alpha.reset(new std::atomic<float>(_trainSettings->alpha));
        
        // NOTE: consider setting size elsewhere
        m_matrixSize = sharedData.trainSettings->size * sharedData.corpus->types.size();
        m_random = sharedData.trainSettings->random;
        
        for (uint8_t i = 0; i < _trainSettings->threads; ++i) {
            m_threads.emplace_back(new trainThread_t(i, sharedData));
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
