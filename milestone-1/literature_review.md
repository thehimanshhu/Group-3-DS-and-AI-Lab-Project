# Milestone 1: Problem Statement and Literature Review

## 1. Project Overview

### 1.1 Project Title

**Deepfake Voice and Scam Detection Using Deep Learning and Generative AI**

### 1.2 Project Summary

With the growth of AI, it has become easier to create fake voices that sound like real people. These voices can be used for impersonation and scams, making it difficult to identify whether an audio recording is genuine or AI-generated.

In this project, we aim to build a system that can identify AI-generated voices and distinguish them from genuine human speech in English, Hindi, Tamil, and Telugu.

Along with voice detection, we will also explore how to identify possible scams by analysing the spoken content, such as requests for money, OTPs, or personal information. We also plan to build a simple web application and Telegram bot where users can submit audio recordings and check the results.

### 1.3 Primary Objective

The main objective is to develop a model that can distinguish genuine human speech from AI-generated speech using a pretrained XLS-R model with a classification layer trained for this task.

We will test the model on held-out audio samples, with particular attention to performance across English, Hindi, Tamil, and Telugu.

## 2. Problem Statement

### 2.1 Problem Definition

One of the main challenges with AI-generated speech is that it can sound very similar to genuine human speech, making it difficult to identify fake audio. This creates a need for a system that can distinguish between genuine and AI-generated voices, including across different languages.

Another challenge is identifying possible scams from spoken content, as suspicious requests may not always be obvious to the listener. This project will address voice authenticity detection as its primary task and explore scam-content detection as an additional component.

### 2.2 Scope of the Project

The project will focus on the following areas:

- Collecting suitable datasets containing genuine human speech and AI-generated speech.
- Developing a model using a pretrained XLS-R model to classify audio as genuine or AI-generated.
- Evaluating the model on audio recordings in English, Hindi, Tamil, and Telugu.
- Exploring NLP techniques to identify possible scam-related content, such as requests for money, OTPs, or personal information.
- Developing a simple web application and Telegram bot to allow users to submit audio recordings and view the results.

### 2.3 Project Boundaries

The primary focus is detecting AI-generated speech. Scam-content detection will be explored separately, as genuine human voices can also be used in scams.

The system will provide predictions based on the submitted audio, but the results may not always be correct. It will therefore serve as a support tool rather than a guarantee that a recording is genuine, fake, safe, or fraudulent.

### 2.4 Measurable Objectives

The main objectives of this project are:

- To develop a model that can distinguish genuine human speech from AI-generated speech.
- To compare the model's performance across English, Hindi, Tamil, and Telugu.
- To evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC.
- To explore NLP techniques to identify possible scam-related content, such as requests for money, OTPs, or personal information.
- To build a basic web application and Telegram bot where users can submit audio recordings and view the results.


## 3. Literature Review and Existing Solutions

### 3.1 Existing Approaches

Existing research on AI-generated voice detection includes deep learning models, speech datasets, and pretrained speech models. The following approaches are relevant to our project.

#### 3.1.1 AASIST (Deep Learning Approach)

**How it works:**

AASIST is a deep learning model designed to distinguish genuine speech from fake or manipulated speech. It analyses patterns in audio to identify differences between real and spoofed voices.

**Strengths:**

- Specifically designed for audio spoofing detection.
- Combines information from different parts of an audio recording.
- Provides an existing model that can be studied as a reference.

**Weaknesses:**

- Its performance may vary when tested on audio from different languages or unseen voice-generation tools.
- It may require further evaluation for noisy or real-world recordings.

**Research reference:** Jung et al. (2022), *AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks*.

#### 3.1.2 Meta Wav2Vec2 XLS-R (300M)

**How it works:**

XLS-R is a pretrained speech model developed by Meta AI. It was trained on approximately 436,000 hours of speech data across 128 languages. It learns patterns from audio recordings and can be adapted for different speech-related tasks.

For our project, we plan to use the pretrained XLS-R model and add a classification layer to distinguish genuine human speech from AI-generated speech. 

**Strengths:**

- Pretrained on speech data from many languages.
- Can be adapted for audio classification using our own dataset.
- Provides a useful starting point for a project involving English, Hindi, Tamil, and Telugu.

**Weaknesses:**

- The model has around 300 million parameters, so training can require significant computing resources.
- It was not originally designed specifically for deepfake voice detection, so it needs task-specific training and evaluation.
- Its performance across our four target languages and different voice-generation tools still needs to be tested.

**Relevance to our project:**

XLS-R can be used as primary model because its multilingual speech representations may help us build a voice detection system for the four target languages. We will need to fine-tune and evaluate it using suitable genuine and AI-generated speech datasets.

**Links:**

- **Pretrained model:** https://huggingface.co/facebook/wav2vec2-xls-r-300m
- **Research paper:** https://arxiv.org/abs/2111.09296
- **Official Meta AI overview:** https://ai.meta.com/blog/xls-r-self-supervised-speech-processing-for-128-languages/
- **Official GitHub repository:** https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr


### 3.2 Research Gaps

From the existing research and datasets, we identified a few areas that need further exploration:

- **Multilingual detection:** We want to evaluate whether a single model can distinguish genuine and AI-generated speech across English, Hindi, Tamil, and Telugu.
- **Different AI voice generators:** The model needs to be tested on speech from different AI tools to understand how well it works on voices it has not encountered during training.
- **Dataset combination:** Genuine and AI-generated speech will come from different sources, so the data needs to be prepared carefully to avoid introducing differences that could affect the model's predictions.
- **Scam-content detection:** Detecting an AI-generated voice does not necessarily mean the recording is a scam. Analysing the spoken content is a separate task that we plan to explore as an additional component.

### 3.3 Proposed Approach

Based on the existing research, we plan to use a pretrained XLS-R model to distinguish genuine human speech from AI-generated speech.

Our proposed approach includes the following steps:

1. **Dataset preparation:** Collect genuine speech samples from datasets such as Kathbath and ASVspoof, and synthetic speech samples from datasets such as IndicSynth and ASVspoof.
2. **Data preprocessing:** Prepare the audio recordings in a consistent format and organise them into genuine and AI-generated speech categories.
3. **Model development:** Use the pretrained XLS-R model with a classification layer and train it on the prepared dataset.
4. **Model evaluation:** Test the model on audio recordings that were not used during training and compare its performance across English, Hindi, Tamil, and Telugu.
5. **Scam-content analysis:** Explore NLP techniques to identify suspicious spoken content, such as requests for money, OTPs, or personal information.
6. **Application development:** Develop a basic web application and Telegram bot to allow users to submit audio recordings and view the analysis results.

We will use the results of the initial experiments to understand the model's strengths and limitations and identify areas for improvement.

### 3.4 Evaluation Metrics

We will use the following metrics to evaluate how well our model distinguishes genuine human speech from AI-generated speech.

- **Accuracy:** Measures the percentage of audio recordings classified correctly.
- **Precision:** Measures how many recordings predicted as AI-generated are actually AI-generated.
- **Recall:** Measures how many of the actual AI-generated recordings are correctly detected.
- **F1-Score:** Provides a balance between precision and recall.
- **ROC-AUC:** Measures how well the model distinguishes between genuine and AI-generated speech across different classification thresholds.

We will compare these metrics across English, Hindi, Tamil, and Telugu to understand how the model performs in each language. We can also examine incorrect predictions to identify areas where the model needs improvement.


## 4. References

1. **AASIST — Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks (2022)**  
   Research paper: https://arxiv.org/abs/2110.01200  
   Official GitHub repository: https://github.com/clovaai/aasist

2. **ASVspoof 2019 — A Large-Scale Public Database of Synthesized, Converted and Replayed Speech (2020)**  
   Research paper: https://arxiv.org/abs/1911.01601  
   Official website: https://www.asvspoof.org/index2019.html  
   Dataset: https://zenodo.org/records/6906306

3. **XLS-R — Self-supervised Cross-lingual Speech Representation Learning at Scale (2021)**  
   Research paper: https://arxiv.org/abs/2111.09296  
   Pretrained model: https://huggingface.co/facebook/wav2vec2-xls-r-300m

6. **XLS-R Model Documentation**  
   Hugging Face model page: https://huggingface.co/facebook/wav2vec2-xls-r-300m  
   Research paper: https://arxiv.org/abs/2111.09296