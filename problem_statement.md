# Problem Statement

Generative AI can now copy a human voice from a few seconds of audio. Open-source voice-cloning models such as XTTS-v2 (Casanova et al., 2024) advertise cloning a voice from "just a 6-second audio clip", and they run on a free GPU. The resulting speech sounds natural, carries the speaker's accent and tone, and is very hard for a listener to tell apart from the real person.

This technology has useful applications, but it is also being misused. Cloned voices are used to:

- Impersonate family members in "emergency" calls asking for money
- Pose as bank officials, police officers or company executives
- Spread fake audio clips of public figures
- Bypass voice-based identity checks

In India, these calls and voice notes arrive in Hindi, Tamil, Telugu and many other languages, not only English. However, almost all published deepfake voice detectors are trained and tested on English speech. Their reliability on Indian languages, on newer voice-generation systems and on low-quality phone audio is largely unknown.

There is a strong need for an accurate, multilingual and explainable AI system that can tell whether a voice is real or AI-generated.
