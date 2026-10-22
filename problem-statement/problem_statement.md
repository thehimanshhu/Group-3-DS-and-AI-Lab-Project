# Problem Statement

> Project Title: Deepfake Voice and Scam Detection Using Deep Learning

AI-generated voices are making phone and voice-message scams harder to spot, especially in multilingual regions like India. Most people cannot tell a cloned voice from a real one, and manual verification is unreliable.

This project aims to build an end-to-end system that detects AI-generated speech and flags likely scams by analyzing both the audio and its spoken content.

Given an audio clip with:

* Speech in English, Hindi, Tamil or Telugu
* Any length, recorded or uploaded
* Clean or phone-quality sound

The system must:

* Predict the probability that the clip is AI-generated
* Highlight the audio segments that influenced the decision
* Stay reliable on languages and voice generators not seen during training

The system also gives a scam verdict, and users can check suspicious calls and voice messages through a simple web app and Telegram bot. Beyond binary classification, it tests how well detection holds up across languages, unseen generators and real-world conditions.
