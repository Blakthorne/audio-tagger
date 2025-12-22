import torch

# Detect and configure Apple Silicon GPU (MPS)
if torch.backends.mps.is_available():
    device = "mps"
    print("🚀 Using Apple Silicon GPU (MPS) for acceleration")
else:
    device = "cpu"
    print("Using CPU (MPS not available)")

# Fix SSL certificate verification issues on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import whisper
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from typing import List

# Import Bible quote processor
from bible_quote_processor import process_text, QuoteBoundary

# Load sentence transformer for semantic paragraph detection
print("Loading semantic model for paragraph detection...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
if device == "mps":
    semantic_model = semantic_model.to(device)
    print("✓ Semantic model loaded on GPU")

def transcribe_audio(file_path: str) -> str:
    print("Transcribing audio...")
    # Use medium model for good speed-accuracy balance
    model = whisper.load_model("medium", device=device)
    
    # Use FP16 on MPS for faster processing
    if device == "mps":
        print("✓ Whisper using Apple Silicon GPU")
    
    # Balanced parameters optimized for long files
    # IMPORTANT: no_speech_threshold is set to None to prevent skipping audio segments.
    # With a threshold like 0.6, Whisper can skip 10-30+ seconds of audio if it detects
    # "silence" (which may actually be soft speech, background music, or pauses).
    # For sermon transcription, we want ALL audio transcribed, even quiet parts.
    result = model.transcribe(
        file_path,
        language="en",  # Specify language for better accuracy
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Temperature fallback to avoid hallucinations
        compression_ratio_threshold=2.4,  # Detect repetitions
        logprob_threshold=-1.0,  # Filter low-confidence segments
        no_speech_threshold=None,  # DISABLED - prevents skipping audio segments
        condition_on_previous_text=True,  # Use context from previous segments
        verbose=True,  # Show progress for long files
        fp16=True,
        initial_prompt="This is a clear audio recording of speech."  # Help model recognize speech
    )
    print("Finished transcribing!")
    return result["text"]

# Prayer detection patterns - sentences that start prayers (NOT including "Amen" which ENDS prayers)
# These patterns should be SPECIFIC to prayer invocations, not just any sentence starting with "God" or "Lord"
PRAYER_START_PATTERNS = [
    r"^let'?s\s+pray",                          # "Let's pray"
    r"^let\s+us\s+pray",                        # "Let us pray"
    r"^dearly?\s+(heavenly\s+)?father",         # "Dear Father", "Dearly Father"  
    r"^dear\s+(lord|god)",                      # "Dear Lord", "Dear God"
    r"^(lord|father|god),?\s+(we|i)\s+(ask|pray|thank|come)\b",  # "Lord, we pray/ask/thank/come"
    r"^in\s+jesus['']?\s+name",                 # "In Jesus' name"
    # NOTE: "Amen" is NOT a prayer start - it ENDS prayers. Handled separately below.
]

# Pattern to detect standalone "Amen" sentences that should be attached to previous paragraph
AMEN_END_PATTERN = r"^amen\s*[.!]?$"


def segment_into_paragraphs(text: str, quote_boundaries: List[QuoteBoundary] = None, 
                            min_sentences_per_paragraph: int = 8, 
                            similarity_threshold: float = 0.65, 
                            window_size: int = 3) -> str:
    """
    Intelligently segment text into paragraphs based on semantic similarity.
    Optimized for rambling speech like sermons - avoids over-segmentation.
    
    IMPORTANT: This function ensures Bible quotes are never split across paragraphs.
    Quotes are treated as atomic units that must stay together.
    
    Also detects prayers and ensures they start new paragraphs.
    
    Args:
        text: The input text to segment (with quotes already marked)
        quote_boundaries: List of QuoteBoundary objects indicating quote positions
                         (used to prevent splitting quotes across paragraphs)
        min_sentences_per_paragraph: Minimum sentences before allowing a paragraph break (default: 8)
        similarity_threshold: Cosine similarity threshold (0-1). Lower = more paragraphs (default: 0.65)
        window_size: Number of sentence transitions to average for smoother detection (default: 3)
    
    Returns:
        Text with paragraph breaks (double newlines)
    """
    print("Segmenting text into paragraphs based on context...")
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if len(sentences) <= min_sentences_per_paragraph:
        return text
    
    # Detect sentences that are prayer starts (should force paragraph break before)
    prayer_start_sentences = set()
    # Detect sentences that are "Amen" (should be attached to previous paragraph)
    amen_sentences = set()
    for sent_idx, sentence in enumerate(sentences):
        sent_stripped = sentence.strip()
        # Check for "Amen" first (takes priority)
        if re.search(AMEN_END_PATTERN, sent_stripped, re.IGNORECASE):
            amen_sentences.add(sent_idx)
        else:
            for pattern in PRAYER_START_PATTERNS:
                if re.search(pattern, sent_stripped, re.IGNORECASE):
                    prayer_start_sentences.add(sent_idx)
                    break
    
    if prayer_start_sentences:
        print(f"   Detected {len(prayer_start_sentences)} prayer start(s) (will force paragraph breaks)")
    if amen_sentences:
        print(f"   Detected {len(amen_sentences)} 'Amen' sentence(s) (will attach to previous paragraph)")
    
    # Build prayer RANGES (start_idx → amen_idx) to prevent breaks within prayers
    # Each PRIMARY prayer start should find its corresponding Amen ending
    # NESTED prayer starts (like "Dearly Father" inside "Let's pray") should NOT force breaks
    sentences_in_prayers = set()
    primary_prayer_starts = set()  # Only these will force paragraph breaks
    prayer_ranges = []  # List of (start, end) tuples
    sorted_prayer_starts = sorted(prayer_start_sentences)
    sorted_amens = sorted(amen_sentences)
    used_amens = set()
    
    for prayer_start_idx in sorted_prayer_starts:
        # Skip if this start is already inside another prayer range
        already_in_range = False
        for (range_start, range_end) in prayer_ranges:
            if range_start <= prayer_start_idx <= range_end:
                already_in_range = True
                break
        
        if already_in_range:
            continue  # This is a nested prayer start, skip it
            
        # Find the first unused Amen after this start
        amen_idx = None
        for candidate_amen in sorted_amens:
            if candidate_amen > prayer_start_idx and candidate_amen not in used_amens:
                amen_idx = candidate_amen
                used_amens.add(amen_idx)
                break
        
        if amen_idx is not None:
            # This is a PRIMARY prayer start - mark it and build range
            primary_prayer_starts.add(prayer_start_idx)
            prayer_ranges.append((prayer_start_idx, amen_idx))
            # Mark all sentences in this range
            for idx in range(prayer_start_idx, amen_idx + 1):
                sentences_in_prayers.add(idx)
    
    if sentences_in_prayers:
        print(f"   {len(sentences_in_prayers)} sentences are within prayers (will not split)")
        if len(primary_prayer_starts) < len(prayer_start_sentences):
            nested = len(prayer_start_sentences) - len(primary_prayer_starts)
            print(f"   {nested} nested prayer pattern(s) detected (will not force breaks)")
    
    # Build a mapping of character positions to sentence indices
    # This helps us identify which sentences are part of quotes
    sentence_char_positions = []
    current_pos = 0
    for sent in sentences:
        start_pos = text.find(sent, current_pos)
        if start_pos == -1:
            start_pos = current_pos
        end_pos = start_pos + len(sent)
        sentence_char_positions.append((start_pos, end_pos))
        current_pos = end_pos
    
    # Identify which sentences are part of quotes (should not be split)
    # For quotes with interjections, we need to track the full quote range
    sentences_in_quotes = set()
    quote_ranges = []  # List of (first_sentence_idx, last_sentence_idx) for each quote
    if quote_boundaries:
        for quote in quote_boundaries:
            first_sent_idx = None
            last_sent_idx = None
            for sent_idx, (sent_start, sent_end) in enumerate(sentence_char_positions):
                # Check if sentence overlaps with quote boundary
                # A sentence is "in" a quote if there's any overlap
                if sent_start < quote.end_pos and sent_end > quote.start_pos:
                    sentences_in_quotes.add(sent_idx)
                    if first_sent_idx is None:
                        first_sent_idx = sent_idx
                    last_sent_idx = sent_idx
            
            # Store the full range for this quote (handles interjections)
            if first_sent_idx is not None and last_sent_idx is not None:
                quote_ranges.append((first_sent_idx, last_sent_idx))
        
        print(f"   {len(sentences_in_quotes)} sentences are within Bible quotes (will not split)")
        if any(end - start > 0 for start, end in quote_ranges):
            multi_sent_quotes = sum(1 for start, end in quote_ranges if end > start)
            print(f"   {multi_sent_quotes} quotes span multiple sentences (will keep together)")
    
    # Get embeddings for all sentences
    print(f"Analyzing {len(sentences)} sentences...")
    embeddings = semantic_model.encode(sentences, convert_to_numpy=True)
    
    # Calculate cosine similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        cos_sim = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        )
        similarities.append(cos_sim)
    
    # Calculate rolling average for smoother topic detection
    smoothed_similarities = []
    for i in range(len(similarities)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(similarities), i + window_size // 2 + 1)
        avg_sim = np.mean(similarities[start_idx:end_idx])
        smoothed_similarities.append(avg_sim)
    
    # Build paragraphs with minimum length requirement
    # CRITICAL: Never break inside a quote (even with interjections)
    # ALSO: Force breaks before AND after prayers (prayers get their own paragraphs)
    paragraphs = []
    current_paragraph = [sentences[0]]
    just_ended_prayer = False  # Track if we just added an Amen
    
    for i, similarity in enumerate(smoothed_similarities):
        next_sentence_idx = i + 1
        
        # If we just ended a prayer with Amen, force a paragraph break now
        if just_ended_prayer:
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            just_ended_prayer = False
        
        # Check if this sentence (i+1) is a PRIMARY prayer start (force break BEFORE it)
        # Only primary prayer starts force breaks - nested ones (like "Dearly Father" inside "Let's pray") don't
        is_new_prayer_start = next_sentence_idx in primary_prayer_starts
        if is_new_prayer_start and current_paragraph:
            # Force paragraph break before prayer
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [sentences[next_sentence_idx]]
            continue
        
        # If this is an "Amen" sentence, add it to current paragraph and mark for break after
        if next_sentence_idx in amen_sentences:
            current_paragraph.append(sentences[next_sentence_idx])
            just_ended_prayer = True  # Will force break on next iteration
            continue
        
        current_paragraph.append(sentences[next_sentence_idx])
        
        # Determine if we CAN break here (not inside a quote OR a prayer)
        can_break = True
        
        # Check if we're in the middle of a prayer - if so, don't break
        if sentences_in_prayers:
            if next_sentence_idx in sentences_in_prayers:
                # Check if the next sentence is also part of the same prayer
                if (next_sentence_idx + 1) < len(sentences) and (next_sentence_idx + 1) in sentences_in_prayers:
                    can_break = False
        
        # Check if we're in the middle of a quote (including quotes with interjections)
        # Use quote_ranges to check if current and next sentence are part of same logical quote
        if can_break and quote_ranges:
            for quote_start, quote_end in quote_ranges:
                # If current sentence is within a quote range and there are more sentences
                # in that same quote range after us, don't break
                if quote_start <= next_sentence_idx <= quote_end:
                    if next_sentence_idx < quote_end:
                        # There are more sentences in this quote - don't break
                        can_break = False
                        break
        
        # Check for interjection pattern: sentence ends with "what?", "who?", etc.
        # and next sentence starts with a quote - these should stay together
        if can_break and (next_sentence_idx + 1) < len(sentences):
            current_sent = sentences[next_sentence_idx].strip()
            following_sent = sentences[next_sentence_idx + 1].strip()
            # Pattern: current ends with interjection question, next starts with quote
            if re.search(r'\b(what|who|where|when|why|how)\?\s*"?\s*$', current_sent, re.IGNORECASE):
                if following_sent.startswith('"'):
                    can_break = False
        
        # Only consider breaking if we have minimum sentences AND we're not in a quote/prayer
        if len(current_paragraph) >= min_sentences_per_paragraph and can_break:
            # Break on significant topic change
            if similarity < similarity_threshold:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Progress indicator for long texts
        if (next_sentence_idx) % 50 == 0:
            print(f"  Processed {next_sentence_idx}/{len(smoothed_similarities)} sentence transitions...")
    
    # Add final paragraph
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    print(f"✓ Created {len(paragraphs)} paragraphs from {len(sentences)} sentences")
    print(f"  Average: {len(sentences) / len(paragraphs):.1f} sentences per paragraph")
    
    # Join paragraphs with double newlines
    return '\n\n'.join(paragraphs)


if __name__ == "__main__":
    import sys
    
    # Get input file from command line or use default
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "20251214-SunAM-Polar.mp3"
    
    # PIPELINE ORDER (optimized):
    # 1. Transcribe audio to raw text
    # 2. Process Bible quotes (auto-detect translation + normalize references + add quotation marks)
    # 3. Segment into paragraphs (respecting quote boundaries)
    
    print("\n" + "=" * 70)
    print("SERMON TRANSCRIPTION PIPELINE")
    print(f"Input: {audio_file}")
    print("Bible Translation: AUTO-DETECT (per-quote)")
    print("=" * 70)
    
    # Step 1: Transcribe audio
    print("\n📝 STEP 1: Transcribing audio...")
    raw = transcribe_audio(audio_file)
    
    # Save raw transcription for debugging
    with open("whisper_raw.txt", "w", encoding="utf-8") as f:
        f.write(raw)
    print("   Raw transcription saved to: whisper_raw.txt")
    
    # Step 2: Process Bible quotes using the bible_quote_processor
    # Translation is auto-detected PER QUOTE from the transcript content
    # This handles speakers who switch translations mid-sermon
    # This normalizes references (e.g., "Hebrews 725" → "Hebrews 7:25")
    # and adds quotation marks around actual Bible quotes
    print("\n📖 STEP 2: Processing Bible quotes (detecting translation per-quote)...")
    with_quotes, quote_boundaries = process_text(raw, translation=None, auto_detect=True, verbose=True)
    
    # Step 3: Segment into paragraphs (respecting quote boundaries)
    # The quote_boundaries are passed so quotes are never split across paragraphs
    print("\n📄 STEP 3: Segmenting into paragraphs...")
    paragraphed = segment_into_paragraphs(
        with_quotes,
        quote_boundaries=quote_boundaries,
        min_sentences_per_paragraph=5,  # At least 5 sentences per paragraph
        similarity_threshold=0.30,  # Break on topic shifts (below mean similarity)
        window_size=3  # Smooth detection over 3 sentence transitions
    )
    
    # Save final output
    with open("whisper_cleaned.txt", "w", encoding="utf-8") as f:
        f.write(paragraphed)
    
    print("\n" + "=" * 70)
    print("✅ TRANSCRIPTION COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  • whisper_raw.txt      - Raw transcription (no processing)")
    print("  • whisper_quotes.txt   - With Bible quotes marked")
    print("  • whisper_cleaned.txt  - Final output with paragraphs")
    print(f"\nPipeline:")
    print("  1. ✓ Audio transcription (Whisper medium model)")
    print("  2. ✓ Bible translation auto-detection (per-quote)")
    print("  3. ✓ Bible quote detection and normalization")
    print("  4. ✓ Paragraph segmentation (quote-aware)")
    print("=" * 70)