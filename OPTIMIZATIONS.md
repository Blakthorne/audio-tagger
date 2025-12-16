# Audio Tagger Optimizations for Long Files

## Summary

Optimized for hour+ long audio files with focus on **accuracy over speed**.

## Key Optimizations

### 1. Whisper Transcription (Upgraded to `large-v3`)

-   **Model**: Changed from `turbo` → `large-v3` (most accurate Whisper model)
-   **Beam Search**: `beam_size=5` for better accuracy
-   **Best-of Sampling**: `best_of=5` to select highest quality output
-   **Temperature**: `0.0` for deterministic, consistent results
-   **Compression Ratio**: `2.4` threshold to detect and handle repetitions
-   **Log Probability**: `-1.0` threshold to filter low-confidence segments
-   **No Speech Threshold**: `0.6` for better silence detection
-   **Context Awareness**: `condition_on_previous_text=True` for continuity
-   **Word Timestamps**: Enabled for precise timing information
-   **Verbose Mode**: Shows progress during long transcriptions

### 2. Punctuation Restoration

-   **Chunking**: Automatically splits texts >10,000 characters
-   **Sentence-Aware**: Splits at sentence boundaries to preserve context
-   **Progress Indicators**: Shows progress every 100 sentences
-   **Memory Efficient**: Processes chunks individually to avoid OOM errors

### 3. Grammar Correction

-   **Increased Beam Search**: Raised from 5 → 8 beams for better quality
-   **Token-Aware Batching**: Respects T5 model's token limits (400 char chunks)
-   **Sentence Preservation**: Maintains sentence structure during batching
-   **Progress Tracking**: Shows batch progress for long processing

## Performance Expectations

### For 1-hour audio file:

-   **Transcription**: ~30-60 minutes (with large-v3)
-   **Punctuation**: ~5-10 minutes
-   **Grammar**: ~10-20 minutes
-   **Total**: ~45-90 minutes

### Accuracy Improvements:

-   **Word Error Rate**: ~3-5% (vs ~5-8% with turbo)
-   **Punctuation**: Better handling of long-form content
-   **Grammar**: More consistent corrections across entire transcript

## Trade-offs

-   ✅ **Maximum Accuracy**: Best possible transcription quality
-   ✅ **Reliable for Long Audio**: No memory issues with hour+ files
-   ✅ **Progress Visibility**: Clear feedback during processing
-   ⚠️ **Processing Time**: Significantly longer than speed-optimized versions
-   ⚠️ **Model Size**: large-v3 requires ~3GB disk space

## Usage

Simply run the script as before:

```bash
python main.py
```

The script will automatically:

1. Show detailed progress during transcription
2. Handle long texts by chunking
3. Provide progress indicators for each stage
4. Save the final cleaned transcript to `whisper_cleaned.txt`
