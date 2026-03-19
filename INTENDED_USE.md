# Intended Use & Copyright Considerations

## What PedalAI Does

PedalAI is a BiLSTM model that predicts sustain pedal timings for piano MIDI files. Given a MIDI sequence of note events (without pedal data), the model outputs a predicted pedal control track, specifically the timing of sustain pedal on/off events.

**This model does not generate music.** It does not produce melodies, harmonies, rhythms, or any other musical content. Its output is limited to pedal timing annotations, a narrow performance parameter that serves to enhance the expressiveness of existing MIDI playback.

## Intended Purpose

PedalAI is intended for:

- Adding realistic sustain pedal behavior to MIDI files that lack pedal data
- Enhancing playback quality in applications where MIDI is performed without human pedal input
- Research and educational exploration of pedal timing patterns in piano performance

PedalAI is **not** intended for:

- Generating, reproducing, or reconstructing copyrighted musical works
- Replacing or competing with human musical performance or composition
- Circumventing copyright protections on musical works

## Relationship to Copyright

### Nature of the task

Under current legal and academic discourse (see [Stober & Dornis, 2026](https://arxiv.org/abs/2502.15858)), a meaningful distinction exists between **generative AI** (models that produce novel expressive content resembling their training data) and **analytical/TDM tasks** (models that extract patterns, classify, or annotate). PedalAI falls into the latter category. The model learns statistical regularities in how pianists use the sustain pedal relative to note patterns. It does not learn or reproduce the musical content itself.

### Training data provenance

PedalAI was trained on pedal and note data extracted from two publicly available datasets:

| Dataset | License | Content | Notes |
|---|---|---|---|
| [POP909](https://github.com/music-x-lab/POP909-Dataset) | MIT (dataset) | MIDI arrangements of 909 Chinese pop songs | The underlying compositions are copyrighted works. The MIT license covers the dataset artifact (MIDI transcriptions and annotations), not the original songs. |
| [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano) | CC BY 4.0 (dataset) | 7,235 classical piano works transcribed from YouTube recordings | Compositions are largely public domain. The CC BY 4.0 license covers the transcribed MIDI data. Source recordings may carry performer's rights in some jurisdictions. |

Neither dataset's license conclusively settles the copyright status of models trained on their contents. This is an unresolved area of law. See the discussion below.

### What the model learns

PedalAI learns the relationship between note event patterns and pedal timing; for example, that pedal changes tend to align with harmonic shifts, phrase boundaries, and bass note movement. This is a statistical pattern that generalizes across pieces, not a reproduction of any specific work's expression.

### Memorization considerations

As with any neural network, some degree of memorization of training examples is possible. If a user provides input that closely matches a training example (e.g., the exact note sequence from a POP909 song minus its pedal track), the model's output may closely resemble the original pedal data for that piece. This is an inherent property of the model architecture, not an intended feature. The output in such cases is limited to pedal timing data, which represents a narrow and largely functionally determined aspect of performance.

## Why MIT

PedalAI exists because other people shared their work openly. POP909 is MIT. GiantMIDI-Piano is CC BY 4.0. The research papers, tools, and libraries this project depends on were made available under permissive terms. Choosing a more restrictive license for a model trained on that openness would be inconsistent.

It is also unclear whether trained model weights are copyrightable in their own right. The weights were not authored; they were computed by an optimization algorithm. Whether they constitute a derivative work of the training data, an original work by the developer, or something not currently covered by copyright law is an open legal question. Given that ambiguity, MIT reflects the honest position: use this freely, attribute it if you find it useful, and make your own judgment about the legal landscape.

## Attribution

If you use PedalAI in your work, please cite the training data sources:

- **POP909**: Wang, Z. et al. "POP909: A Pop-song Dataset for Music Arrangement Generation." ISMIR, 2020. (MIT License)
- **GiantMIDI-Piano**: Kong, Q. et al. "GiantMIDI-Piano: A large-scale MIDI dataset for classical piano music." 2020. (CC BY 4.0)

GiantMIDI-Piano's CC BY 4.0 license requires attribution. This notice satisfies that requirement.

## Disclaimer

This document represents the author's good-faith understanding of the relevant legal and ethical considerations as of March 2026. It is not legal advice. Copyright law as it applies to machine learning (including the questions of whether model training constitutes fair use or TDM, whether trained weights are derivative works, and who holds rights over model parameters) is unsettled and actively evolving. Users are responsible for ensuring their use of PedalAI complies with applicable laws in their jurisdiction.
