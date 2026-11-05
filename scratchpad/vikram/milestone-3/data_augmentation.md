Implement the following audio data augmentation techniques in the training pipeline:

1. **Noise:** Add background noise with an SNR of approximately 15 dB.
2. **SpecAugment:** Apply 2 time masks and 2 frequency masks to the audio spectrogram.
3. **Shift, Speed, and Pitch:** Apply random time shifts (±0.15 seconds), speed changes (0.9–1.1×), and pitch shifts (±2 semitones).
4. **Reverberation:** Add room reverberation with an RT60 of approximately 0.3 seconds.

**Requirements:**
- Apply augmentations randomly during training only.
- Apply them to both real and fake audio with the same probabilities.
- Keep the original labels unchanged.
- Make augmentation probabilities and parameters configurable.
- Preserve audio compatibility with the existing model pipeline.
- Avoid modifying the original dataset files.
- Keep the implementation modular, clean, and consistent with the existing project structure.
