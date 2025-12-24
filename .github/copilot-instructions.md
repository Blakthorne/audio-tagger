# Sermon Transcription Pipeline

## Project Overview

This project transcribes sermon audio files and intelligently formats the output by detecting Bible quotes, normalizing references, and segmenting text into coherent paragraphs. The pipeline uses OpenAI Whisper for transcription, semantic embeddings for paragraph detection, and the Bolls.life Bible API for quote verification.

## Core Architecture

### Pipeline Flow (main.py)

1. **Audio Transcription** → Whisper medium model (GPU-accelerated on Apple Silicon MPS)
2. **Bible Quote Processing** → Reference normalization + quote boundary detection + auto-translation detection
3. **Paragraph Segmentation** → Semantic similarity analysis (respects quote/prayer boundaries)

**Output Files:**

-   `whisper_raw.txt` - Raw transcription (no processing)
-   `whisper_quotes.txt` - Intermediate with Bible quotes marked (if generated)
-   `whisper_cleaned.txt` - Final output with proper paragraphs

### Key Components

**main.py**

-   `transcribe_audio()` - Whisper transcription with critical parameter: `no_speech_threshold=None` prevents skipping quiet segments (10-30+ seconds of audio can be lost otherwise)
-   `segment_into_paragraphs()` - Creates paragraphs using semantic embeddings while NEVER splitting Bible quotes or prayers

**bible_quote_processor.py** (3000+ lines)

-   `process_text()` - Main entry point for Bible quote processing
-   Reference normalization: Fixes transcription errors like "Hebrews 725" → "Hebrews 7:25"
-   Auto-translation detection: Analyzes quote text to determine KJV vs NIV vs ESV per-quote
-   Fuzzy matching: Matches transcript text to Bible API verses (threshold: 0.60-0.70)
-   Interjection detection: Handles mid-quote interruptions like "a what?" or "who?"
-   Uses Bolls.life API with local caching (`bible_verse_cache.json`)

## Critical Development Rules

### Paragraph Segmentation Logic

-   **Quote boundaries are SACRED** - Never split a quote across paragraphs, even with interjections
-   **Prayer detection** - Prayers get their own paragraphs (start patterns in `PRAYER_START_PATTERNS`, end with "Amen")
-   **Minimum paragraph length** - Default 5-8 sentences prevents over-segmentation for rambling speech
-   The `quote_boundaries` parameter from `process_text()` MUST be passed to `segment_into_paragraphs()`

### Whisper Transcription Settings

DO NOT change these without understanding consequences:

-   `no_speech_threshold=None` - Disabling prevents skipping audio (default 0.6 causes major data loss)
-   `temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` - Fallback temperatures prevent hallucinations
-   `condition_on_previous_text=True` - Uses context from previous segments
-   `fp16=True` - Required for Apple Silicon GPU acceleration

### Bible Quote Processing

-   Translation auto-detection (per-quote): Speaker may switch translations mid-sermon
-   API calls are cached to `bible_verse_cache.json` - never delete this during processing
-   Rate limiting: 0.5s delay between API calls (Bolls.life API)
-   Fuzzy matching thresholds: 0.60 general, 0.70 for quote starts (tuned for sermon audio)

## Developer Workflows

### Run the Pipeline

```bash
python3 main.py [audio_file.mp3]
# Default: 20251214-SunAM-Polar.mp3
```

### Apple Silicon GPU Detection

The code auto-detects and uses Apple Silicon GPU (MPS) for acceleration:

-   Whisper transcription uses MPS device
-   Semantic model (sentence-transformers) uses MPS device
-   Check console output for "🚀 Using Apple Silicon GPU (MPS)" confirmation

### Testing Bible Quote Detection

Run standalone processor (includes verification):

```bash
python3 bible_quote_processor.py whisper_raw.txt
```

### Agent Testing Mode Requirement

-   When the agent (you) is testing parts of the pipeline other than the audio transcription itself, use the `test` mode of the main script. Invoke:

```bash
python3 main.py test
```

This causes `main.py` to load `whisper_test.txt` as the raw transcription and bypass the Whisper transcription step, allowing faster iteration when validating quote processing, paragraph segmentation, and other post-transcription logic.

### Installing Dependencies

```bash
pip install -r requirements.txt
```

Note: `torch` must support Apple Silicon MPS for GPU acceleration

## File Conventions

### Data Structures

-   `QuoteBoundary` (bible_quote_processor.py) - Stores quote positions in text (start_pos, end_pos, reference, translation)
-   `BibleReference` - Parsed Bible reference (book, chapter, verse_start, verse_end)

### Key Patterns

-   Bible books: `BIBLE_BOOKS` dict maps variations (e.g., "gen", "genesis") to canonical names
-   Book ID mapping: `BOOK_ID_MAP` maps canonical names to Bolls.life API IDs (1-66)
-   Interjection patterns: `INTERJECTION_PATTERNS` regex list for mid-quote interruptions
-   Prayer patterns: `PRAYER_START_PATTERNS` for detecting prayer invocations (NOT "Amen" which ends prayers)

### SSL Certificate Workaround

macOS SSL verification is disabled at startup:

```python
ssl._create_default_https_context = ssl._create_unverified_context
```

This is required for Bible API calls on some macOS configurations.

## Common Pitfalls

1. **Don't split quotes** - Always pass `quote_boundaries` to paragraph segmentation
2. **Don't skip audio** - Keep `no_speech_threshold=None` in Whisper params
3. **Don't guess translations** - Use auto-detection or verify API response
4. **Don't break prayers** - Respect `PRAYER_START_PATTERNS` and "Amen" detection
5. **Don't ignore cache** - Respect `bible_verse_cache.json` to avoid redundant API calls

## Integration Points

-   **Whisper API** - OpenAI Whisper (local model, no external API)
-   **Bolls.life Bible API** - Free, no API key required, 50+ translations
-   **Sentence Transformers** - 'all-MiniLM-L6-v2' model for semantic similarity
-   **PyTorch MPS** - Apple Silicon GPU acceleration (auto-detected)
