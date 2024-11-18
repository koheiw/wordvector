/**
 * @file
 * @brief trainThread trains a word2vec model from the specified part of train data set file
 * @author Max Fomichev
 * @date 20.12.2016
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/

#include "trainThread.hpp"

namespace w2v {
    trainThread_t::trainThread_t(uint8_t _id, const sharedData_t &_sharedData) :
            m_sharedData(_sharedData), m_randomGenerator(m_sharedData.trainSettings->random),
            m_rndWindowShift(0, static_cast<short>((m_sharedData.trainSettings->window - 1))),
            m_downSampling(), m_nsDistribution(), m_hiddenLayerVals(), m_hiddenLayerErrors(),
            m_thread() {

        if (!m_sharedData.trainSettings) {
            throw std::runtime_error("train settings are not initialized");
        }

        if (m_sharedData.trainSettings->sample > 0.0f) {
            m_downSampling.reset(new downSampling_t(m_sharedData.trainSettings->sample,
                                                    m_sharedData.corpus->trainWords));
        }

        if (m_sharedData.trainSettings->negative > 0) {
            m_nsDistribution.reset(new nsDistribution_t(m_sharedData.corpus->frequency));
        }

        if (m_sharedData.trainSettings->withHS && !m_sharedData.huffmanTree) {
            throw std::runtime_error("Huffman tree object is not initialized");
        }

        m_hiddenLayerErrors.reset(new std::vector<float>(m_sharedData.trainSettings->size));
        if (m_sharedData.trainSettings->algorithm == 1) {
            m_hiddenLayerVals.reset(new std::vector<float>(m_sharedData.trainSettings->size));
        }

        if (!m_sharedData.corpus) {
            throw std::runtime_error("corpus object is not initialized");
        }
        
        // NOTE: specify range for workers
        auto n = m_sharedData.corpus->texts.size();
        auto threads = m_sharedData.trainSettings->threads;
        range = std::make_pair(floor((n / threads) * _id),
                               floor((n / threads) * (_id + 1)) - 1);
        
    }

    void trainThread_t::worker(std::vector<float> &_trainMatrix) noexcept {
        
        for (auto g = 1; g <= m_sharedData.trainSettings->iterations; ++g) {
            
            std::size_t threadProcessedWords = 0;
            std::size_t prvThreadProcessedWords = 0;
            
            // for progressCallback
            auto wordsPerAllThreads = m_sharedData.trainSettings->iterations * m_sharedData.corpus->trainWords;
            auto wordsPerAlpha = wordsPerAllThreads / 10000;
            
            float alpha = 0;
            for (std::size_t h = range.first; h <= range.second; ++h) {

                // calculate alpha
                if (threadProcessedWords - prvThreadProcessedWords > wordsPerAlpha) { // next 0.01% processed
                    *m_sharedData.processedWords += threadProcessedWords - prvThreadProcessedWords;
                    prvThreadProcessedWords = threadProcessedWords;

                    float ratio = static_cast<float>(*(m_sharedData.processedWords)) / wordsPerAllThreads;
                    alpha = m_sharedData.trainSettings->alpha * (1 - ratio);
                    if (alpha < m_sharedData.trainSettings->alpha * 0.0001f) {
                        alpha = m_sharedData.trainSettings->alpha * 0.0001f;
                    }
                    (*m_sharedData.alpha) = alpha;
                }
                
                text_t text = m_sharedData.corpus->texts[h];
                
                // read sentence
                std::vector<unsigned int> sentence;
                sentence.reserve(text.size());
                for (size_t i = 0; i < text.size(); ++i) {

                    auto &word = text[i];
                    // ignore padding
                    if (word == 0) { 
                        continue; 
                    }
                    // ignore infrequent words
                    if (m_sharedData.corpus->frequency[word - 1] < m_sharedData.trainSettings->minWordFreq) {
                        continue;
                    }
                    
                    threadProcessedWords++;
                    if (m_sharedData.trainSettings->sample > 0.0f) {
                        if ((*m_downSampling)(m_sharedData.corpus->frequency[word - 1], m_randomGenerator)) {
                            continue; // skip this word
                        }
                    }
                    sentence.push_back(word - 1); // zero-based index of words
                }
                
                if (m_sharedData.trainSettings->algorithm == 1) {
                    cbow(sentence, _trainMatrix);
                } else if (m_sharedData.trainSettings->algorithm == 2) {
                    skipGram(sentence, _trainMatrix);
                }
            }
            // print progress
            if (m_sharedData.progressCallback != nullptr) {
                m_sharedData.progressCallback(g, alpha);
            }
        }
    }

    inline void trainThread_t::cbow(const std::vector<unsigned int> &_text,
                                    std::vector<float> &_trainMatrix) noexcept {
        
        // NOTE: define K = m_sharedData.trainSettings->size here
        std::size_t K = m_sharedData.trainSettings->size;
        if (_text.size() == 0)
            return;
        for (std::size_t i = 0; i < _text.size(); ++i) {
            // hidden layers initialized with 0 values
            std::memset(m_hiddenLayerVals->data(), 0, m_hiddenLayerVals->size() * sizeof(float));
            std::memset(m_hiddenLayerErrors->data(), 0, m_hiddenLayerErrors->size() * sizeof(float));
            
            // NOTE: consider generating posRndWindow as j
            //       check how downsampling is implemented
            auto rndShift = m_rndWindowShift(m_randomGenerator);
            rndShift = 0; // disable random shift
            //std::cout << rndShift << "\n";
            std::size_t cw = 0;
            // NOTE: define token = _text[posRndWindow] here;
            //int skip = 0; // for develpment
            for (auto j = rndShift; j < m_sharedData.trainSettings->window * 2 + 1 - rndShift; ++j) {
                if (j == m_sharedData.trainSettings->window) {
                    //skip++;
                    continue;
                }

                auto posRndWindow = i - m_sharedData.trainSettings->window + j;
                // std::cout << "i = " <<  i << ",";
                // std::cout << " posRndWindow = " <<  posRndWindow << "\n";
                if (posRndWindow >= _text.size()) {
                    //skip++;
                    continue;
                }
    
                for (std::size_t k = 0; k < K; ++k) {
                    // (*m_hiddenLayerVals)[k] += _trainMatrix[k + _text[posRndWindow]
                    //                                        * m_sharedData.trainSettings->size];
                    (*m_hiddenLayerVals)[k] += _trainMatrix[k + _text[posRndWindow] * K];
                }
                cw++;
                // std::cout << "i = " <<  i << ", ";
                // std::cout << "posRndWindow = " <<  posRndWindow << ", ";
                // std::cout << "_text.size() = " <<  _text.size() << "\n";
            }
            
            //std::cout << "skip = " <<  skip << "\n";
            if (cw == 0) {
                continue;
            }
            // NOTE: j should be k
            //for (std::size_t j = 0; j < m_sharedData.trainSettings->size; j++) {
            for (std::size_t k = 0; k < K; k++) {
                (*m_hiddenLayerVals)[k] /= cw;
            }
            
            if (m_sharedData.trainSettings->withHS) {
                hierarchicalSoftmax(_text[i], *m_hiddenLayerErrors, *m_hiddenLayerVals, 0);
            } else {
                negativeSampling(_text[i], *m_hiddenLayerErrors, *m_hiddenLayerVals, 0);
            }
            
            // hidden -> in
            for (auto j = rndShift; j < m_sharedData.trainSettings->window * 2 + 1 - rndShift; ++j) {
                if (j == m_sharedData.trainSettings->window) {
                    continue;
                }

                auto posRndWindow = i - m_sharedData.trainSettings->window + j;
                if (posRndWindow >= _text.size()) {
                    continue;
                }
                //for (std::size_t k = 0; k < m_sharedData.trainSettings->size; ++k) {
                for (std::size_t k = 0; k < K; ++k) {
                    _trainMatrix[k + _text[posRndWindow] * K] += (*m_hiddenLayerErrors)[k];
                }
            }
        }
    }

    inline void trainThread_t::skipGram(const std::vector<unsigned int> &_text,
                                        std::vector<float> &_trainMatrix) noexcept {
        if (_text.size() == 0)
            return;
        for (std::size_t i = 0; i < _text.size(); ++i) {
            auto rndShift = m_rndWindowShift(m_randomGenerator);
            rndShift = 0;
            for (auto j = rndShift; j < m_sharedData.trainSettings->window * 2 + 1 - rndShift; ++j) {
                if (j == m_sharedData.trainSettings->window) {
                    continue;
                }

                auto posRndWindow = i - m_sharedData.trainSettings->window + j;
                if (posRndWindow >= _text.size()) {
                    continue;
                }
                // shift to the selected word vector in the matrix
                auto shift = _text[posRndWindow] * m_sharedData.trainSettings->size;

                // hidden layer initialized with 0 values
                std::memset(m_hiddenLayerErrors->data(), 0, m_hiddenLayerErrors->size() * sizeof(float));

                if (m_sharedData.trainSettings->withHS) {
                    hierarchicalSoftmax(_text[i], (*m_hiddenLayerErrors), _trainMatrix, shift);
                } else {
                    negativeSampling(_text[i], (*m_hiddenLayerErrors), _trainMatrix, shift);
                }

                for (std::size_t k = 0; k < m_sharedData.trainSettings->size; ++k) {
                    _trainMatrix[k + shift] += (*m_hiddenLayerErrors)[k];
                }
            }
        }
    }

    inline void trainThread_t::hierarchicalSoftmax(std::size_t _index,
                                                   std::vector<float> &_hiddenLayer,
                                                   std::vector<float> &_trainLayer,
                                                   std::size_t _trainLayerShift) noexcept {
        auto huffmanData = m_sharedData.huffmanTree->huffmanData(_index);
        for (std::size_t i = 0; i < huffmanData->huffmanCode.size(); ++i) {
            auto l2 = huffmanData->huffmanPoint[i] * m_sharedData.trainSettings->size;
            // Propagate hidden -> output
            float f = 0.0f;
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                f += _trainLayer[j + _trainLayerShift] * (*m_sharedData.bpWeights)[j + l2];
            }
            if (f < -m_sharedData.trainSettings->expValueMax) {
//            f = 0.0f;
                continue; // original approach
            } else if (f > m_sharedData.trainSettings->expValueMax) {
//            f = 1.0f;
                continue; // original approach
            } else {
                f = (*m_sharedData.expTable)[static_cast<std::size_t>((f + m_sharedData.trainSettings->expValueMax)
                                                                      * (m_sharedData.expTable->size()
                                                                         / m_sharedData.trainSettings->expValueMax /
                                                                         2))];
            }

            auto gradientXalpha = (1.0f - static_cast<float>(huffmanData->huffmanCode[i]) - f) * (*m_sharedData.alpha);
            // Propagate errors output -> hidden
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                _hiddenLayer[j] += gradientXalpha * (*m_sharedData.bpWeights)[j + l2];
            }
            // Learn weights hidden -> output
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                (*m_sharedData.bpWeights)[j + l2] += gradientXalpha * _trainLayer[j + _trainLayerShift];
            }
        }
    }

    inline void trainThread_t::negativeSampling(std::size_t _index,
                                                std::vector<float> &_hiddenLayer,
                                                std::vector<float> &_trainLayer,
                                                std::size_t _trainLayerShift) noexcept {
        for (std::size_t i = 0; i < static_cast<std::size_t>(m_sharedData.trainSettings->negative) + 1; ++i) {
            std::size_t target = 0;
            bool label = false;
            if (i == 0) {
                target = _index;
                label = true;
            } else {
                target = (*m_nsDistribution)(m_randomGenerator);
                if (target == _index) {
                    continue;
                }
            }

            auto l2 = target * m_sharedData.trainSettings->size;
            // Propagate hidden -> output
            float f = 0.0f;
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                f += _trainLayer[j + _trainLayerShift] * (*m_sharedData.bpWeights)[j + l2];
            }
            if (f < -m_sharedData.trainSettings->expValueMax) {
                f = 0.0f;  // original approach
//            continue;
            } else if (f > m_sharedData.trainSettings->expValueMax) {
                f = 1.0f;  // original approach
//            continue;
            } else {
                f = (*m_sharedData.expTable)[static_cast<std::size_t>((f + m_sharedData.trainSettings->expValueMax)
                                                                      * (m_sharedData.expTable->size()
                                                                         / m_sharedData.trainSettings->expValueMax /
                                                                         2))];
            }

            auto gradientXalpha = (static_cast<float>(label) - f) * (*m_sharedData.alpha);
            // Propagate errors output -> hidden
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                _hiddenLayer[j] += gradientXalpha * (*m_sharedData.bpWeights)[j + l2];
            }
            // Learn weights hidden -> output
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                (*m_sharedData.bpWeights)[j + l2] += gradientXalpha * _trainLayer[j + _trainLayerShift];
            }
        }
    }
}
