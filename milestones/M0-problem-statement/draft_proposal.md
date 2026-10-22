# Deepfake Voice Detection for Indian Languages - Project Proposal

---

## 1. Introduction

The rapid advancement of Generative AI has made voice cloning cheap and widely accessible. Open-source models can now replicate a person's voice from only a few seconds of audio. This has enabled a new wave of fraud, where scammers use cloned voices to impersonate family members, bank officials or government authorities in phone calls and voice messages.

In India, these scams reach people in many languages, including Hindi, Tamil and Telugu. However, most existing deepfake voice detectors are trained and tested mainly on English speech, and their reliability on Indian languages is largely unknown. As a result, there is a critical need for an intelligent, multilingual and explainable AI-driven solution capable of detecting AI-generated speech with high accuracy.

---

### In the News

I'll search for documented cases and reports. Here are the cases and reports I found, newest first, followed by the surveys and trend reports.

## Individual case reports

1. Mumbai senior citizen, 69, loses ₹50 lakh in an AI voice-cloning scam where the caller mimicked a respected community figure (Oct 2026): https://www.freepressjournal.in/mumbai/mumbai-cyber-fraud-69-year-old-senior-citizen-loses-50-lakh-in-ai-voice-cloning-scam

2. Indore resident loses ₹1.83 lakh after an AI-generated voice posed as his brother-in-law in Sydney, claiming he faced jail over an expired visa (Jul 2026): https://incidentdatabase.ai/cite/1566/

3. Indore play school owner loses ₹97,500, her savings and teachers' salary money, to a cloned voice of her police-officer cousin; police call it Madhya Pradesh's first such case (Jan 2026): https://incidentdatabase.ai/reports/7000/

4. Powai businessman loses ₹80,000 after fraudsters played his son's cloned voice pleading for bail, posing as the Indian Embassy in Dubai (2024): https://incidentdatabase.ai/reports/5140

5. Delhi and Noida parents get calls with cloned "child's voice" crying for help, one threatening a false rape case against an MBA student: https://www.tabla.com.sg/news/india/scammers-use-ai-to-clone-kids-voices
   - The snippet I retrieved didn't show a publication year, so check the date on the page.

## Surveys and trend reports

6. McAfee report finds 83% of Indian AI voice-scam victims lost money, and 69% of Indians can't tell a cloned voice from a real one: https://www.boomlive.in/amp/explainers/ai-voice-scam-voice-cloning-mcafee-report-21851

7. About 47% of Indian adults have experienced or know someone who experienced an AI voice scam; tools can clone a voice from three seconds of audio: https://businessworld.in/article/ai-voice-scams-more-prevalent-in-india-affects-almost-50-indians-mcafee-report-475192

8. Aggregator summary reports a 450% rise in voice-cloning fraud complaints to Haryana's cyber cell in Q4 2025, and ₹22,846 crore in cyber-fraud losses in 2024: https://threatcluster.io/cluster/ai-voice-cloning-scams-surge-in-india-124ab5b6
   - This is a secondary aggregator, so verify the figures against the original police or news sources before citing them.

---

## 2. Problem Statement

The objective of this project is to design and develop an end-to-end Deep Learning-based system capable of identifying AI-generated (deepfake) speech in English and a few Indian languages.

Given an audio clip consisting of:

* Speech in English, Hindi, Tamil or Telugu
* Recorded or uploaded audio of any length
* Clean or phone-quality recordings

The system must:

* Predict the probability of the clip being AI-generated.
* Highlight the parts of the audio that influenced the decision.
* Remain reliable on languages and voice-generation systems it has not seen during training.

The system moves beyond simple binary classification by testing how well detection holds up across languages, unseen generators and real-world conditions.

---

## 3. Motivation

Current deepfake voice detection systems largely rely on:

* English-only training data
* Older voice-generation systems
* Clean, studio-quality recordings

These approaches fail to generalize to Indian languages, newer voice-cloning tools and noisy phone audio. Additionally, most existing detectors operate as black boxes, offering little interpretability to users.

An intelligent, multilingual and explainable framework can:

* Help reduce financial and emotional harm from voice-cloning scams.
* Show users why a clip was flagged, which builds trust in the result.

---

## 4. Proposed Solution

The proposed system integrates three major AI components:

### 4.1 Generative AI-Based Dataset Creation

Open-source text-to-speech and voice-cloning models (e.g., AI4Bharat Indic TTS, XTTS or similar) will be used to generate:

* Fake speech in Hindi, Tamil and Telugu
* Cloned voices of real speakers
* Samples from multiple different generators

These are combined with real speech from open corpora (e.g., Kathbath, ASVspoof) to create a balanced multilingual dataset.

### 4.2 Deep Learning-Based Detection

A pre-trained multilingual speech model (e.g., wav2vec2-xls-r-300m or similar architecture) will be fine-tuned to analyze:

* Acoustic artifacts
* Unnatural prosody and pitch patterns
* Generator-specific signatures

The model outputs a probability score for the clip being AI-generated. A spectrogram-based CNN will serve as a baseline for comparison.

### 4.3 Explainable Output and Deployment

An explanation module will highlight which segments of the audio drove the decision. The system will be made available through:

* A web application where users upload or record audio
* A WhatsApp bot where users forward suspicious voice notes

Each result includes a real/fake verdict, a confidence score and a visual explanation. This enhances transparency and user trust.

---

## 5. System Architecture Overview

### High-Level Flow

| Component | Function | Output |
| :--- | :--- | :--- |
| **Input** | Audio clip (uploaded, recorded or forwarded) | Raw audio |
| **Preprocessing** | Standardizes and segments the audio | Clean audio segments |
| **Speech Encoder** | Extracts multilingual speech features | Feature representations |
| **Classifier** | Predicts real vs AI-generated | Probability score |
| **Score Aggregation** | Combines scores across segments | Final deepfake score |
| **Explanation Module** | Highlights influential audio segments | Visual explanation |
| **Final Report** | Presents prediction + explanation via web app or WhatsApp | Final outcome |

The architecture ensures both predictive accuracy and interpretability.

---

## 6. Expected Outcomes

The project aims to deliver:

* A high-accuracy deepfake voice detection model for English and Indian languages
* A new multilingual dataset of real and AI-generated Indian-language speech
* An evaluation of how well detection transfers across languages and unseen voice generators
* Explainable AI outputs for end users
* A deployable prototype accessible through a web app and WhatsApp

---

## 7. Novelty of the Approach

The novelty of the proposed system lies in:

* Building a new Indian-language deepfake speech dataset using recent Generative AI voice models
* Testing detection across languages and unseen voice generators, which existing work rarely does
* Evaluating robustness on phone-quality audio, matching how scam calls actually reach people
* Providing explainable, accessible results through a web app and a WhatsApp bot
