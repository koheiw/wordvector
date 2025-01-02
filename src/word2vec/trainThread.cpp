/**
 * @file
 * @brief trainThread trains a word2vec model from the specified part of train data set file
 * @author Max Fomichev
 * @date 20.12.2016
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/

#include "trainThread.hpp"

namespace w2v {
    // NOTE: make m_rndWindow
    trainThread_t::trainThread_t(uint16_t _number, const data_t &_data) :
            m_number(_number), m_data(_data), m_randomGenerator(m_data.settings->random),
            m_rndWindowShift(0, static_cast<short>((m_data.settings->window - 1))), // NOTE: to delete
            m_rndWindow(1, static_cast<short>((m_data.settings->window))), // NOTE: added
            m_downSampling(), m_nsDistribution(), m_hiddenLayerVals(), m_hiddenLayerErrors(),
            m_thread() {

        if (!m_data.settings) {
            throw std::runtime_error("train settings are not initialized");
        }

        if (m_data.settings->sample < 1.0f) {
            m_downSampling.reset(new downSampling_t(m_data.settings->sample,
                                                    m_data.corpus->trainWords));
        }

        if (m_data.settings->negative > 0) {
            m_nsDistribution.reset(new nsDistribution_t(m_data.corpus->frequency));
        }

        if (m_data.settings->withHS && !m_data.huffmanTree) {
            throw std::runtime_error("Huffman tree object is not initialized");
        }

        m_hiddenLayerErrors.reset(new std::vector<float>(m_data.settings->size));
        m_hiddenLayerVals.reset(new std::vector<float>(m_data.settings->size)); // not used in SG
        
        if (!m_data.corpus) {
            throw std::runtime_error("corpus object is not initialized");
        }
        
        // NOTE: specify range for workers
        auto n = m_data.corpus->texts.size();
        auto threads = m_data.settings->threads;
        range = std::make_pair(floor((n / threads) * m_number),
                               floor((n / threads) * (m_number + 1)) - 1);
        
    }

    void trainThread_t::worker(std::vector<float> &_trainMatrix, int &_iter, float &_alpha) noexcept {
        
        for (auto g = 1; g <= m_data.settings->iterations; ++g) {
            
            std::size_t threadProcessedWords = 0;
            std::size_t prvThreadProcessedWords = 0;
            
            auto wordsPerAllThreads = m_data.settings->iterations * m_data.corpus->trainWords;
            auto wordsPerAlpha = wordsPerAllThreads / 10000;
            
            //std::cout << "type = " << m_data.settings->type << "\n";
            //std::cout << "minWordFreq = " << m_data.settings->minWordFreq << "\n";
            float alpha = 0;
            for (std::size_t h = range.first; h <= range.second; ++h) {
                
                // calculate alpha
                if (threadProcessedWords - prvThreadProcessedWords > wordsPerAlpha) { // next 0.01% processed
                    *m_data.processedWords += threadProcessedWords - prvThreadProcessedWords;
                    prvThreadProcessedWords = threadProcessedWords;

                    float ratio = static_cast<float>(*(m_data.processedWords)) / wordsPerAllThreads;
                    alpha = m_data.settings->alpha * (1 - ratio);
                    if (alpha < m_data.settings->alpha * 0.0001f) {
                        alpha = m_data.settings->alpha * 0.0001f;
                    }
                    (*m_data.alpha) = alpha;
                }
                
                text_t text = m_data.corpus->texts[h];
                //std::cout << "text = " <<  text.size() << "\n";
                
                // read sentence
                std::vector<unsigned int> sentence;
                sentence.reserve(text.size());
                for (size_t i = 0; i < text.size(); ++i) {

                    auto &word = text[i];
                    // ignore padding
                    if (word == 0) { 
                        //std::cout << "padding: " << word << "\n";
                        continue; 
                    }
                    // ignore infrequent words
                    if (m_data.corpus->frequency[word - 1] < m_data.settings->minWordFreq) {
                        //std::cout << "infrequent: " << word << "\n";
                        continue;
                    }
                    
                    threadProcessedWords++;
                    if (m_data.settings->sample < 1.0f) {
                        if ((*m_downSampling)(m_data.corpus->frequency[word - 1], m_randomGenerator)) {
                            //std::cout << "downsample: " << word << "\n";
                            continue; // skip this word
                        }
                    }
                    sentence.push_back(word - 1); // zero-based index of words
                }
                
                //std::cout << "sentence = " <<  sentence.size() << "\n";
                if (m_data.settings->type == 1) {
                    cbow(sentence, _trainMatrix);
                } else if (m_data.settings->type == 2) {
                    skipGram(sentence, _trainMatrix);
                }
            }
            // for progress message
            if (m_number == 0) {
                _iter = g;
                _alpha = alpha;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    inline void trainThread_t::cbow(const std::vector<unsigned int> &_text,
                                    std::vector<float> &_trainMatrix) noexcept {
        
        std::size_t K = m_data.settings->size;
        if (_text.size() == 0)
            return;
        for (std::size_t i = 0; i < _text.size(); ++i) {
            // hidden layers initialized with 0 values
            std::memset(m_hiddenLayerVals->data(), 0, m_hiddenLayerVals->size() * sizeof(float));
            std::memset(m_hiddenLayerErrors->data(), 0, m_hiddenLayerErrors->size() * sizeof(float));

            int window = m_rndWindow(m_randomGenerator);
            std::size_t from = std::max(0, (int)i - window);
            std::size_t to = std::min((int)_text.size(), (int)i + window);
            std::size_t cw = 0;
            for (std::size_t j = from; j < to; ++j) {
                if (j == i)
                    continue;
                for (std::size_t k = 0; k < K; ++k) {
                    (*m_hiddenLayerVals)[k] += _trainMatrix[k + _text[j] * K];
                }
                cw++;
            }
            
            if (cw == 0)
                continue;
            for (std::size_t k = 0; k < K; ++k) {
                (*m_hiddenLayerVals)[k] /= cw;
            }
            
            if (m_data.settings->withHS) {
                hierarchicalSoftmax(_text[i], *m_hiddenLayerErrors, *m_hiddenLayerVals, 0);
            } else {
                negativeSampling(_text[i], *m_hiddenLayerErrors, *m_hiddenLayerVals, 0);
            }
            
            // hidden -> in
            for (std::size_t j = from; j < to; ++j) {
                if (j == i)
                    continue;
                for (std::size_t k = 0; k < K; ++k) {
                    _trainMatrix[k + _text[j] * K] += (*m_hiddenLayerErrors)[k];
                }
            }
        }
    }
    
    inline void trainThread_t::skipGram(const std::vector<unsigned int> &_text,
                                        std::vector<float> &_trainMatrix) noexcept {
        
        std::size_t K = m_data.settings->size;
        if (_text.size() == 0)
            return;
        for (std::size_t i = 0; i < _text.size(); ++i) {
            int window = m_rndWindow(m_randomGenerator);
            std::size_t from = std::max(0, (int)i - window);
            std::size_t to = std::min((int)_text.size(), (int)i + window);
            for (std::size_t j = from; j < to; j++) {
                if (j == i)
                    continue;
                
                // hidden layer initialized with 0 values
                std::memset(m_hiddenLayerErrors->data(), 0, m_hiddenLayerErrors->size() * sizeof(float));
                
                // shift to the selected word vector in the matrix
                auto shift = _text[j] * K;
                if (m_data.settings->withHS) {
                    hierarchicalSoftmax(_text[i], (*m_hiddenLayerErrors), _trainMatrix, shift);
                } else {
                    negativeSampling(_text[i], (*m_hiddenLayerErrors), _trainMatrix, shift);
                }
                for (std::size_t k = 0; k < m_data.settings->size; ++k) {
                    _trainMatrix[k + shift] += (*m_hiddenLayerErrors)[k];
                }
            }
        }
    }

    inline void trainThread_t::hierarchicalSoftmax(std::size_t _index,
                                                   std::vector<float> &_hiddenLayer,
                                                   std::vector<float> &_trainLayer,
                                                   std::size_t _trainLayerShift) noexcept {
        
        std::size_t K = m_data.settings->size;
        auto huffmanData = m_data.huffmanTree->huffmanData(_index);
        for (std::size_t i = 0; i < huffmanData->huffmanCode.size(); ++i) {
            auto shift = huffmanData->huffmanPoint[i] * K;
            
            // propagate hidden -> output
            float f = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                f += _trainLayer[k + _trainLayerShift] * (*m_data.bpWeights)[k + shift];
            }
            float prob = 0;
            if (f < -m_data.settings->expValueMax) {
                //continue;
                prob = 0.0f;
            } else if (f > m_data.settings->expValueMax) {
                //continue;
                prob = 1.0f;
            } else {
                auto r = (f + m_data.settings->expValueMax) * (m_data.expTable->size() / m_data.settings->expValueMax / 2);
                prob = (*m_data.expTable)[static_cast<std::size_t>(r)];
            }
            
            // compute gradient x alpha
            auto gxa = (1.0f - static_cast<float>(huffmanData->huffmanCode[i]) - prob) * (*m_data.alpha);
            // propagate errors output -> hidden
            for (std::size_t k = 0; k < K; ++k) {
                _hiddenLayer[k] += gxa * (*m_data.bpWeights)[k + shift];
            }
            // learn weights hidden -> output
            for (std::size_t k = 0; k < K; ++k) {
                (*m_data.bpWeights)[k + shift] += gxa * _trainLayer[k + _trainLayerShift];
            }
        }
    }

    inline void trainThread_t::negativeSampling(std::size_t _index,
                                                std::vector<float> &_hiddenLayer,
                                                std::vector<float> &_trainLayer,
                                                std::size_t _trainLayerShift) noexcept {
        
        std::size_t K = m_data.settings->size;
        for (std::size_t i = 0; i < static_cast<std::size_t>(m_data.settings->negative) + 1; ++i) {
            std::size_t target = 0;
            bool label = false;
            if (i == 0) {
                // positive case
                target = _index;
                label = true;
            } else {
                // negative case
                target = (*m_nsDistribution)(m_randomGenerator);
                if (target == _index) {
                    continue;
                }
            }
            auto shift = target * K;
            
            // propagate hidden -> output
            float f = 0.0f;
            // predict likelihood of _index using logistic regression
            for (std::size_t k = 0; k < K; ++k) {
                f += _trainLayer[k + _trainLayerShift] * (*m_data.bpWeights)[k + shift];
            }
            //std::cout << f << "\n";
            float prob = 0;
            if (f < -m_data.settings->expValueMax) {
                prob = 0.0f;
            } else if (f > m_data.settings->expValueMax) {
                prob = 1.0f;
            } else {
                auto r = (f + m_data.settings->expValueMax) * (m_data.expTable->size() / m_data.settings->expValueMax / 2);
                prob = (*m_data.expTable)[static_cast<std::size_t>(r)];
            }
            
            // compute gradient x alpha
            auto gxa = (static_cast<float>(label) - prob) * (*m_data.alpha); // gxa > 0 in the positive case
            //std::cout << i << ": " << _index << ", " <<  target << ", " << gxa << "\n";
            // propagate errors output -> hidden
            for (std::size_t k = 0; k < K; ++k) {
                _hiddenLayer[k] += gxa * (*m_data.bpWeights)[k + shift]; // added to _trainMatrix
            }
            // learn weights hidden -> output
            for (std::size_t k = 0; k < m_data.settings->size; ++k) {
                (*m_data.bpWeights)[k + shift] += gxa * _trainLayer[k + _trainLayerShift];
            }
        }
    }
}
