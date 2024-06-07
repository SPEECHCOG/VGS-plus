# VGS_seq_or_parallel_CogSci

This repository contains the instructions and scripts to replicate experiments reported in:

Khorrami, Cruz Blandon & Räsänen: Computational Insights to Acquisition of Phonemes, Words, and Word Meanings in Early Language: Sequential or Parallel Acquisition? Proc. CogSci-2023, Sydney, Australia. (https://escholarship.org/uc/item/79t028n8)

# Model Source

This project's model is based on the work from the following repository:

https://github.com/jasonppy/FaST-VGS-Family
 
To train the model, please download the code from the above repository and follow the provided instructions. Additionally, please ensure that you give credit to the creators for their contributions to the model.

For setting the weight of the audio (SSL) and audio-visual (VGS) losses (i.e., alpha coefficient), you need to modify the "weight_loss" function within the "steps/trainer.py" file of the model's source code. 

# Model Description

The VGS+ model combines a wav2vec 2.0-based speech self-supervised learning (SSL) and a transformer-based visually grounded speech (VGS) learning mechanisms within one model. It has shown that the speech representations obtained from hidden layers of the trained VGS+ model contain phonemic and lexical information. 

# How to Use

## Phoneme discrimination score

"abx.py" provides the speech representations of a given hidden layer (of the speech encoder and decoder) for ABX task. Please first modify the path to the input data (test audio files) and the path to save the output data (speech embeddings). 

For measuring the ABX phoneme discrimination score, please follow the instructions in following repository:

https://github.com/zerospeech/zerospeech2021

You can use the template provided at "abx.sh" to obtain ABX score for different layers (0:11) of any specific model (specified by the path to the bundle file of the model). 

For test data, you need to download dev-clean subset of LibriSpeech data from https://www.openslr.org/12 .

## Lexicon discrimination score

"lexical.py" provides the speech representations of a given hidden layer (of the speech encoder and decoder) for lexical task. Please first modify the path to the input data (test audio files) and the path to save the output data (speech embeddings).

For measuring the lexical score, please follow the instruction in the following repository:

https://github.com/SPEECHCOG/CDI_lextest

You can use the template provided at "lexica.sh" to obtain lexical score for different layers (0:11) of any specific model (specified by the path to the bundle file of the model). 
