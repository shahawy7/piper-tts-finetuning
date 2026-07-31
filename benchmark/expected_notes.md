# Benchmark Listening Evaluation Rubric

This document defines the evaluation criteria and expected phonological metrics for benchmark sentences when testing baseline vs. fine-tuned Piper Arabic models.

---

## 🎧 Evaluation Criteria (Scale 1 - 5)

| Metric | Score 1 (Poor) | Score 3 (Acceptable) | Score 5 (Excellent) |
|---|---|---|---|
| **Diacritic Pronunciation (Tashkeel)** | Frequent mispronunciations of short vowels (Fatha, Damma, Kasra) and Sukun. | Correct short vowels on most words, occasional glitches on Tanween/Shaddah. | Flawless rendering of all diacritics, Tanween, and Shaddah. |
| **Phoneme Accuracy (Makharij)** | Unclear emphatic sounds (ص, ض, ط, ظ, ع, ح). Distortion on glottal stops (Hamza). | Clear consonants with minor softness on emphatic letters. | Crisp, distinct articulation of all Arabic phonemes (ح، خ، ع، غ، ص، ض، ط، ظ). |
| **Prosody & Naturalness** | Robotic, monotone, jerky rhythm, unnatural pauses. | Moderately natural speech flow with acceptable cadence. | Human-like intonation, smooth phrasing, natural pitch contours. |
| **Audio Quality & Clarity** | Muffled, robotic buzzing, clipping, or background noise. | Clear audio, negligible artifacts. | Studio quality, 22,050 Hz clean waveform without metallic distortion. |
| **Real-Time Factor (RTF)** | RTF > 1.0 (Slower than real-time synthesis). | 0.2 < RTF <= 0.5. | RTF <= 0.2 (High-speed real-time synthesis). |

---

## 📝 Sentence Evaluation Focus

1. **Sentence 1**: Greeting phrase (`السَّلَامُ عَلَيْكُمْ`). Test for natural pauses and Shaddah on `س`.
2. **Sentence 2**: Technical vocabulary (`التَّحْوِيلِ الصَّوْتِيِّ`). Test for consecutive Shaddah and Ya (`يِّ`).
3. **Sentence 3**: Idiomatic MSA (`الْعِلْمُ نُورٌ`). Test for Tanween (`نُورٌ`) and contrast between `ع` and `ج`.
4. **Sentence 4**: Speed & clarity (`بِدِقَّةِ النُّطْقِ وَسُرْعَةِ الْأَدَاءِ`). Test for `ق`, `ط`, and final Hamza (`ءِ`).
5. **Sentence 5**: Interrogative intonation (`هَلْ يُمْكِنُ...؟`). Test for question pitch inflection and long sentence breath control.
6. **Sentence 6**: Proper nouns (`عَمَّانَ`, `الْأُرْدُنِيَّةِ`). Test for correct stress on geographical names.
7. **Sentence 7**: Numbers and countable noun agreement (`ثَلَاثَةَ`, `خَمْسَ`). Test for numeral pronunciation and Kasratain.
8. **Sentence 8**: Descriptive narrative (`تُشْرِقُ الشَّمْسُ`). Test for solar letters (`الرّ، الشّ`) and soft vowels.
9. **Sentence 9**: Target task phrase (`تَحْسِينِ نُطْقِ الْكَلِمَاتِ الْمُشَكَّلَةِ`). Primary indicator of fine-tuning improvement.
10. **Sentence 10**: Abstract nouns (`الصَّبْرَ وَالِاسْتِمْرَارِيَّةَ`). Test for hamzat al-wasl (`وَالِاسْتِمْرَارِيَّةَ`) and long words.
