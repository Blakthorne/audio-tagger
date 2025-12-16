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

def segment_into_paragraphs(text: str, min_sentences_per_paragraph: int = 8, similarity_threshold: float = 0.65, window_size: int = 3) -> str:
    """
    Intelligently segment text into paragraphs based on semantic similarity.
    Optimized for rambling speech like sermons - avoids over-segmentation.
    
    Args:
        text: The input text to segment
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
    paragraphs = []
    current_paragraph = [sentences[0]]
    
    for i, similarity in enumerate(smoothed_similarities):
        current_paragraph.append(sentences[i + 1])
        
        # Only consider breaking if we have minimum sentences
        if len(current_paragraph) >= min_sentences_per_paragraph:
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
    raw = transcribe_audio("20251214-SunAM-Polar.mp3")
    
    # Segment into paragraphs optimized for rambling speech (sermons)
    # min_sentences_per_paragraph: 8-12 works well for sermons
    # Lower similarity_threshold (0.6-0.65) for more breaks, higher (0.7-0.75) for fewer
    paragraphed = segment_into_paragraphs(
        raw, 
        min_sentences_per_paragraph=10,  # At least 10 sentences per paragraph
        similarity_threshold=0.65,  # Only break on significant topic shifts
        window_size=3  # Smooth detection over 3 sentence transitions
    )

    with open("whisper_cleaned.txt", "w", encoding="utf-8") as f:
        f.write(paragraphed)
    
    print("\n✓ Transcription complete with paragraph segmentation!")
    print(f"✓ Output saved to: whisper_cleaned.txt")