#### 1. ASVspoof (Dataset and Benchmark)

**How it works:**

ASVspoof is a research initiative that provides datasets and benchmarks for detecting fake or manipulated speech. The ASVspoof 2019 dataset contains genuine speech, synthesised speech, converted speech, and replayed recordings. Researchers use it to train and evaluate voice detection systems.

**Strengths:**

- Contains genuine and fake speech samples.
- Covers different types of speech spoofing.
- Provides a common benchmark for comparing detection models.

**Weaknesses:**

- The dataset mainly uses English speech and does not cover all four languages in our project.
- Models trained on it may need further testing on audio generated using newer AI tools.
- Some recordings may not fully represent real-world calls and voice messages.

**Relevance to our project:**

We can use ASVspoof as a potential dataset for training or evaluating our model. Additional datasets will be needed to support our focus on Hindi, Tamil, and Telugu.

**Links:**

- Official dataset website: https://www.asvspoof.org/index2019.html
- Research paper: https://doi.org/10.1016/j.csl.2020.101114

#### 2. IndicSynth (Synthetic Speech Dataset)

**How it works:**

IndicSynth is a dataset developed by researchers at IIIT-Delhi for studying AI-generated speech and audio deepfake detection. It contains around 4,000 hours of synthetic speech across 12 Indian languages, including Hindi, Tamil, and Telugu. The dataset was introduced in a research paper published at ACL 2025. 

**Strengths:**

- Covers 12 Indian languages, making it useful for multilingual voice detection research.
- Contains a large amount of AI-generated speech.
- Includes information about the speakers and the models used to generate the audio.

**Weaknesses:**

- It mainly provides synthetic speech, so genuine speech samples will also be needed to train a model to distinguish real from fake voices.
- Its coverage of Indian languages does not automatically guarantee that a model will perform equally well across all languages.
- The dataset's access conditions and licence need to be checked before use.

**Relevance to our project:**

IndicSynth could help us collect AI-generated speech for Hindi, Tamil, and Telugu. We can combine suitable samples from this dataset with genuine speech from other datasets to train and evaluate our XLS-R-based model.

**Links:**

- **Dataset:** https://huggingface.co/datasets/vdivyasharma/IndicSynth
- **Research paper:** https://aclanthology.org/2025.acl-long.1070/
- **Dataset information:** https://aikosh.indiaai.gov.in/home/datasets/details/indicsynth.html

#### 3. Kathbath (Indian Language Speech Dataset)

**How it works:**

Kathbath is a speech dataset developed by AI4Bharat as part of the IndicSUPERB project. It contains around 1,684 hours of labelled human speech across 12 Indian languages, including Hindi, Tamil, and Telugu. It was mainly created for speech recognition research, but it may also be useful for our project as a source of genuine speech recordings.

**Strengths:**

- Provides genuine human speech in all three of our target Indian languages.
- Contains a large amount of speech data from different speakers.
- Includes audio recordings and their corresponding text transcripts.

**Weaknesses:**

- It contains genuine speech rather than AI-generated speech, so fake speech samples must be obtained separately.
- It was designed mainly for speech recognition, not specifically for deepfake detection.
- Access to the files requires accepting the dataset's conditions on Hugging Face.

**Relevance to our project:**

We can use Kathbath as a source of genuine speech and combine suitable samples with AI-generated speech from IndicSynth. This will help us prepare a dataset for training and evaluating our XLS-R-based voice detection model.

**Links:**

- **Dataset:** https://huggingface.co/datasets/ai4bharat/Kathbath
- **Official GitHub repository:** https://github.com/AI4Bharat/IndicSUPERB
- **Research paper:** https://arxiv.org/abs/2208.11761



