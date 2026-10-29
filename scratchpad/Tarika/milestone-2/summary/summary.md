# Generated speech dataset: summary

Source: `TarikaHedaoo/dsai-group3-fake-speech` (manifests). Created by `summarize_dataset.py`.

## Checks

| check                                                         | result                                                           | ok   |
|:--------------------------------------------------------------|:-----------------------------------------------------------------|:-----|
| Expected clips (sentences.csv)                                | 6480                                                             | yes  |
| Clips in manifests                                            | 6480                                                             | yes  |
| Missing clips                                                 | 0                                                                | yes  |
| Unexpected clips                                              | 0                                                                | yes  |
| Duplicate clip_id                                             | 0                                                                | yes  |
| Failed clips (no audio)                                       | 0                                                                | yes  |
| Duplicate audio files (same sha1)                             | 0                                                                | yes  |
| Missing manifest columns                                      | none                                                             | yes  |
| Empty values in required columns                              | 0                                                                | yes  |
| Clips whose text/language/generator differ from sentences.csv | 0                                                                | yes  |
| Sentences spoken by all 3 generators                          | 2160 of 2160                                                     | yes  |
| One sample rate per generator                                 | indic-parler-tts: 44100 Hz, indicf5: 24000 Hz, mms-tts: 16000 Hz | yes  |
| Labels                                                        | fake                                                             | yes  |
| Splits                                                        | test                                                             | yes  |
| Clips flagged silent                                          | 1                                                                | yes  |
| Clips flagged too short                                       | 0                                                                | yes  |
| Clips with > 1% clipped samples                               | 0                                                                | yes  |

## Clips per generator and language

| generator        | language   |   clips |   failed |   flagged |   hours |   dur_min_s |   dur_median_s |   dur_max_s |   sample_rate |   voices |   female |   male |   gender_unknown |
|:-----------------|:-----------|--------:|---------:|----------:|--------:|------------:|---------------:|------------:|--------------:|---------:|---------:|-------:|-----------------:|
| Indic Parler-TTS | Hindi      |     720 |        0 |         0 |    0.99 |         2.2 |            4.8 |        10.2 |         44100 |        4 |      360 |    360 |                0 |
| Indic Parler-TTS | Tamil      |     720 |        0 |         0 |    1.41 |         3.3 |            6.8 |        15.5 |         44100 |        3 |      480 |    240 |                0 |
| Indic Parler-TTS | Telugu     |     720 |        0 |         1 |    1.37 |         3.4 |            6.6 |        14.4 |         44100 |        4 |      360 |    360 |                0 |
| Indic Parler-TTS | All        |    2160 |        0 |         1 |    3.76 |         2.2 |            6   |        15.5 |         44100 |        7 |     1200 |    960 |                0 |
| IndicF5          | Hindi      |     720 |        0 |         0 |    1.04 |         1.9 |            5   |        10.3 |         24000 |       10 |      360 |    360 |                0 |
| IndicF5          | Tamil      |     720 |        0 |         0 |    0.98 |         2   |            4.6 |        10.8 |         24000 |       10 |      360 |    360 |                0 |
| IndicF5          | Telugu     |     720 |        0 |         0 |    1.36 |         2.8 |            6.4 |        19.4 |         24000 |       10 |      360 |    360 |                0 |
| IndicF5          | All        |    2160 |        0 |         0 |    3.38 |         1.9 |            5.3 |        19.4 |         24000 |       30 |     1080 |   1080 |                0 |
| MMS-TTS          | Hindi      |     720 |        0 |         0 |    0.98 |         2.7 |            4.8 |         8.6 |         16000 |        1 |        0 |      0 |              720 |
| MMS-TTS          | Tamil      |     720 |        0 |         0 |    1.61 |         4   |            7.9 |        17.3 |         16000 |        1 |        0 |      0 |              720 |
| MMS-TTS          | Telugu     |     720 |        0 |         0 |    1.52 |         4   |            7.4 |        13.7 |         16000 |        1 |        0 |      0 |              720 |
| MMS-TTS          | All        |    2160 |        0 |         0 |    4.11 |         2.7 |            6.6 |        17.3 |         16000 |        3 |        0 |      0 |             2160 |
| All              | All        |    6480 |        0 |         1 |   11.25 |         1.9 |            5.9 |        19.4 |               |       40 |     2280 |   2040 |             2160 |

## Input sentences

| language   |   sentences |   kathbath_speakers |   words_min |   words_median |   words_max |   female_reader_share |
|:-----------|------------:|--------------------:|------------:|---------------:|------------:|----------------------:|
| Hindi      |         720 |                  20 |           8 |             12 |          16 |                  0.5  |
| Tamil      |         720 |                  20 |           6 |             10 |          15 |                  0.57 |
| Telugu     |         720 |                  20 |           8 |             10 |          15 |                  0.51 |

## Voices

| generator        | lang   | voice_id              | voice_gender   |   clips |
|:-----------------|:-------|:----------------------|:---------------|--------:|
| Indic Parler-TTS | Hindi  | parler-divya          | f              |     180 |
| Indic Parler-TTS | Hindi  | parler-rohit          | m              |     180 |
| Indic Parler-TTS | Hindi  | parler-unnamed-f      | f              |     180 |
| Indic Parler-TTS | Hindi  | parler-unnamed-m      | m              |     180 |
| Indic Parler-TTS | Tamil  | parler-jaya           | f              |     240 |
| Indic Parler-TTS | Tamil  | parler-unnamed-f      | f              |     240 |
| Indic Parler-TTS | Tamil  | parler-unnamed-m      | m              |     240 |
| Indic Parler-TTS | Telugu | parler-lalitha        | f              |     180 |
| Indic Parler-TTS | Telugu | parler-prakash        | m              |     180 |
| Indic Parler-TTS | Telugu | parler-unnamed-f      | f              |     180 |
| Indic Parler-TTS | Telugu | parler-unnamed-m      | m              |     180 |
| IndicF5          | Hindi  | ivr-S4256820200338001 | f              |      72 |
| IndicF5          | Hindi  | ivr-S4256942400332561 | f              |      72 |
| IndicF5          | Hindi  | ivr-S4257120600367211 | f              |      72 |
| IndicF5          | Hindi  | ivr-S4257196900365559 | m              |      72 |
| IndicF5          | Hindi  | ivr-S4257878300397507 | f              |      72 |
| IndicF5          | Hindi  | ivr-S4258354000329548 | m              |      72 |
| IndicF5          | Hindi  | ivr-S4258388600389373 | m              |      72 |
| IndicF5          | Hindi  | ivr-S4258486700370849 | m              |      72 |
| IndicF5          | Hindi  | ivr-S4258561600395718 | f              |      72 |
| IndicF5          | Hindi  | ivr-S4258858600387188 | m              |      72 |
| IndicF5          | Tamil  | ivr-S4256620800339114 | m              |      72 |
| IndicF5          | Tamil  | ivr-S4256700400339825 | f              |      72 |
| IndicF5          | Tamil  | ivr-S4256730700334621 | f              |      72 |
| IndicF5          | Tamil  | ivr-S4256817200333561 | f              |      72 |
| IndicF5          | Tamil  | ivr-S4257156700348229 | f              |      72 |
| IndicF5          | Tamil  | ivr-S4257394700369395 | f              |      72 |
| IndicF5          | Tamil  | ivr-S4258842100383591 | m              |      72 |
| IndicF5          | Tamil  | ivr-S4259410500335240 | m              |      72 |
| IndicF5          | Tamil  | ivr-S4259519400365771 | m              |      72 |
| IndicF5          | Tamil  | ivr-S4259592300370250 | m              |      72 |
| IndicF5          | Telugu | ivr-S4256002300334350 | m              |      72 |
| IndicF5          | Telugu | ivr-S4256004500332662 | m              |      72 |
| IndicF5          | Telugu | ivr-S4256021800330914 | m              |      72 |
| IndicF5          | Telugu | ivr-S4256036600335426 | f              |      72 |
| IndicF5          | Telugu | ivr-S4257679000359335 | f              |      72 |
| IndicF5          | Telugu | ivr-S4257970400395219 | f              |      72 |
| IndicF5          | Telugu | ivr-S4258241700338477 | m              |      72 |
| IndicF5          | Telugu | ivr-S4259164200355294 | f              |      72 |
| IndicF5          | Telugu | ivr-S4259385700302647 | f              |      72 |
| IndicF5          | Telugu | ivr-S4259410400337776 | m              |      72 |
| MMS-TTS          | Hindi  | mms-hi                | unknown        |     720 |
| MMS-TTS          | Tamil  | mms-ta                | unknown        |     720 |
| MMS-TTS          | Telugu | mms-te                | unknown        |     720 |

## Figures

- `fig_duration.png`: clip duration per generator and language
- `fig_speaking_rate.png`: words per second per generator and language
- `fig_loudness.png`: RMS level (dBFS) per generator and language
- `fig_gender.png`: voice gender per generator and language
- `fig_sentence_length.png`: words per input sentence
