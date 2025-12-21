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
    # Post-processing handles punctuation, grammar, etc.
    model = whisper.load_model("medium", device=device)
    
    # Use FP16 on MPS for faster processing
    if device == "mps":
        print("✓ Whisper using Apple Silicon GPU")
    
    # Balanced parameters optimized for long files
    # Note: Using fp16=False on MPS to avoid hallucination issues (exclamation marks)
    result = model.transcribe(
        file_path,
        language="en",  # Specify language for better accuracy
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Temperature fallback to avoid hallucinations
        compression_ratio_threshold=2.4,  # Detect repetitions
        logprob_threshold=-1.0,  # Filter low-confidence segments
        no_speech_threshold=0.6,  # Better silence detection
        condition_on_previous_text=True,  # Use context from previous segments
        verbose=True,  # Show progress for long files
        fp16=True,
        initial_prompt="This is a clear audio recording of speech. You know how to notate Bible references."  # Help model recognize speech
    )
    print("Finished transcribing!")
    return result["text"]


def segment_into_paragraphs(text: str, quote_boundaries: List[QuoteBoundary] = None, 
                            min_sentences_per_paragraph: int = 8, 
                            similarity_threshold: float = 0.65, 
                            window_size: int = 3) -> str:
    """
    Intelligently segment text into paragraphs based on semantic similarity.
    Optimized for rambling speech like sermons - avoids over-segmentation.
    
    IMPORTANT: This function ensures Bible quotes are never split across paragraphs.
    Quotes are treated as atomic units that must stay together.
    
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
    sentences_in_quotes = set()
    if quote_boundaries:
        for quote in quote_boundaries:
            for sent_idx, (sent_start, sent_end) in enumerate(sentence_char_positions):
                # Check if sentence overlaps with quote boundary
                # A sentence is "in" a quote if there's any overlap
                if sent_start < quote.end_pos and sent_end > quote.start_pos:
                    sentences_in_quotes.add(sent_idx)
        
        print(f"   {len(sentences_in_quotes)} sentences are within Bible quotes (will not split)")
    
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
    # CRITICAL: Never break inside a quote
    paragraphs = []
    current_paragraph = [sentences[0]]
    
    for i, similarity in enumerate(smoothed_similarities):
        current_paragraph.append(sentences[i + 1])
        
        # Determine if we CAN break here (not inside a quote)
        # We can only break AFTER sentence i+1 if NEITHER sentence i+1 NOR sentence i+2
        # are in the same quote as any previous sentence in this potential break point
        can_break = True
        
        # Check if we're in the middle of a quote
        if sentences_in_quotes:
            # If current sentence (i+1) is in a quote, check if next sentence is also in the same quote
            if (i + 1) in sentences_in_quotes:
                # Find which quote this sentence belongs to
                for quote in quote_boundaries:
                    sent_start, sent_end = sentence_char_positions[i + 1]
                    if sent_start < quote.end_pos and sent_end > quote.start_pos:
                        # This sentence is in a quote - check if the quote continues
                        # Don't break if the next sentence is also in this quote
                        if (i + 2) < len(sentences):
                            next_sent_start, next_sent_end = sentence_char_positions[i + 2]
                            if next_sent_start < quote.end_pos:
                                can_break = False
                                break
        
        # Only consider breaking if we have minimum sentences AND we're not in a quote
        if len(current_paragraph) >= min_sentences_per_paragraph and can_break:
            # Break on significant topic change
            if similarity < similarity_threshold:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Progress indicator for long texts
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(smoothed_similarities)} sentence transitions...")
    
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
    
    # Save quote-processed version for debugging
    with open("whisper_quotes.txt", "w", encoding="utf-8") as f:
        f.write(with_quotes)
    print("   Quote-processed text saved to: whisper_quotes.txt")
    
    # Step 3: Segment into paragraphs (respecting quote boundaries)
    # The quote_boundaries are passed so quotes are never split across paragraphs
    print("\n📄 STEP 3: Segmenting into paragraphs...")
    paragraphed = segment_into_paragraphs(
        with_quotes,
        quote_boundaries=quote_boundaries,
        min_sentences_per_paragraph=10,  # At least 10 sentences per paragraph
        similarity_threshold=0.65,  # Only break on significant topic shifts
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