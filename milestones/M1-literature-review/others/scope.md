# Scope and Boundaries

## In Scope

- Detecting AI-generated (deepfake) speech in English, Hindi, Tamil and Telugu
- Building a dataset of real speech and AI-generated voices from multiple generators, including clean and phone-quality audio
- Training deep learning models for real vs. fake classification with a probability score
- Highlighting the audio segments that most influenced the prediction
- Evaluating generalization to unseen languages, unseen voice generators and noisy or compressed audio
- Delivering a simple web application and Telegram bot for uploading audio and viewing results (fake probability, highlighted segments, scam verdict)
- Transcribing speech and using NLP to flag scam patterns (requests for money, OTPs, personal information, urgent payments)

## Out of Scope

- Languages beyond English, Hindi, Tamil and Telugu
- Real-time detection during live phone calls, and integration with telecom networks or dialer apps
- Video deepfakes and lip-sync analysis
- Identifying or verifying a specific speaker, or tracing which generator produced a clip
- Blocking calls, reporting scammers or taking any automated action against them
- Production-grade deployment, large-scale infrastructure and mobile apps
- Legal or forensic certification of results
