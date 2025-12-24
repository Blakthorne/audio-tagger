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
from typing import List, Optional

# Import KeyBERT for keyword extraction (tags)
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("⚠️  KeyBERT not installed. Install with: pip install keybert")

# Import Bible quote processor
from bible_quote_processor import process_text, QuoteBoundary

# Load sentence transformer for semantic paragraph detection
print("Loading semantic model for paragraph detection...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
if device == "mps":
    semantic_model = semantic_model.to(device)
    print("✓ Semantic model loaded on GPU")

# High-quality model for tag extraction (loaded lazily when needed)
TAG_MODEL_NAME = "all-mpnet-base-v2"  # Highest quality sentence-transformers model
tag_model = None  # Loaded on first use

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

# Pattern to detect sentences that END with "Amen" (prayer endings)
# This matches "Amen.", "In Jesus' name we pray, Amen.", etc.
AMEN_END_PATTERN = r"\bamen\s*[.!]?\s*$"


def convert_to_markdown(transcript: str, quote_boundaries: List[QuoteBoundary] = None,
                        tags: List[str] = None, scripture_refs: List[str] = None) -> str:
    """
    Convert the final transcript to a formatted markdown file.
    
    Output structure:
    1. Tags section (first)
    2. Scripture References section (second)
    3. Transcript with formatting:
       - Bible quotes (text in "...") are italicized
       - Bible verse references (e.g., "Matthew 2:1-12") are bolded
    
    Args:
        transcript: The paragraphed transcript text
        quote_boundaries: List of QuoteBoundary objects for quotes
        tags: List of keyword tag strings
        scripture_refs: List of scripture reference strings
    
    Returns:
        Formatted markdown string
    """
    print("Converting to markdown format...")
    
    markdown_parts = []
    
    # Section 1: Tags (if available)
    if tags:
        markdown_parts.append("## Tags\n")
        markdown_parts.append(", ".join(tags))
        markdown_parts.append("\n")
    
    # Section 2: Scripture References (if available)
    if scripture_refs:
        markdown_parts.append("\n---\n")
        markdown_parts.append("\n## Scripture References\n")
        markdown_parts.append("\n".join(f"- {ref}" for ref in scripture_refs))
        markdown_parts.append("\n")
    
    # Section 3: Transcript with formatting
    if tags or scripture_refs:
        markdown_parts.append("\n---\n")
    markdown_parts.append("\n## Transcript\n\n")
    
    # Process the transcript to add formatting
    formatted_transcript = transcript
    
    # Remove any existing metadata sections from the transcript (they'll be at the start now)
    # These are added at the end in the current pipeline, so strip them
    if "---\n\n## Scripture References" in formatted_transcript:
        formatted_transcript = formatted_transcript.split("---\n\n## Scripture References")[0].strip()
    if "---\n\n## Tags" in formatted_transcript:
        formatted_transcript = formatted_transcript.split("---\n\n## Tags")[0].strip()
    
    # Step A: Italicize Bible quotes (text within quotation marks that are actual Bible quotes)
    # We need to find quoted text that matches quote boundaries
    if quote_boundaries:
        # Build a set of quote texts for quick lookup
        quote_texts = set()
        for qb in quote_boundaries:
            quote_texts.add(qb.verse_text.strip().lower())
        
        # Find all quoted text in the transcript and italicize Bible quotes
        # Pattern matches text within "..." or "..."
        def italicize_quote(match):
            full_match = match.group(0)
            inner_text = match.group(1)
            
            # Check if this is a Bible quote by comparing with known quote texts
            inner_lower = inner_text.strip().lower()
            
            # Check for partial match (quote might be part of the text)
            is_bible_quote = False
            for qt in quote_texts:
                # Check if there's significant overlap
                if qt in inner_lower or inner_lower in qt:
                    is_bible_quote = True
                    break
                # Check for substantial word overlap (for quotes with slight variations)
                qt_words = set(qt.split())
                inner_words = set(inner_lower.split())
                if len(qt_words & inner_words) >= min(3, len(qt_words)):
                    is_bible_quote = True
                    break
            
            if is_bible_quote:
                # Return italicized version (keep the quotes, add asterisks)
                return f'*"{inner_text}"*'
            return full_match
        
        # Match both regular quotes and smart quotes
        quote_pattern = r'"([^"]+)"'
        formatted_transcript = re.sub(quote_pattern, italicize_quote, formatted_transcript)
    
    # Step B: Bold Bible verse references in the text
    # Create a pattern that matches Bible references like "Matthew 2:1-12", "John 3:16", etc.
    # This should match the book names followed by chapter:verse patterns
    bible_books_pattern = r'\b(' + '|'.join([
        # Old Testament
        'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
        'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings',
        '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther',
        'Job', 'Psalms?', 'Proverbs', 'Ecclesiastes', 'Song of Solomon',
        'Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel',
        'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah',
        'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi',
        # New Testament
        'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans',
        '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians',
        'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians',
        '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews',
        'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John',
        'Jude', 'Revelation'
    ]) + r')\s+(\d+)(?::(\d+)(?:-(\d+))?)?'
    
    def bold_reference(match):
        full_match = match.group(0)
        # Only bold if it looks like a proper reference (has chapter at minimum)
        return f'**{full_match}**'
    
    formatted_transcript = re.sub(bible_books_pattern, bold_reference, formatted_transcript, flags=re.IGNORECASE)
    
    # Clean up any double-bolding that might occur
    formatted_transcript = re.sub(r'\*\*\*\*', '**', formatted_transcript)
    
    markdown_parts.append(formatted_transcript)
    
    result = "".join(markdown_parts)
    
    print(f"   ✓ Markdown conversion complete")
    if quote_boundaries:
        print(f"      • {len(quote_boundaries)} quotes italicized")
    if scripture_refs:
        print(f"      • {len(scripture_refs)} references section")
    if tags:
        print(f"      • {len(tags)} tags in header")
    
    return result


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


# Curated vocabulary of valid religious/theological tags for Christian/Baptist/Evangelical sermons
# These are the ONLY words that can become tags - ensures religious relevance
RELIGIOUS_TAG_VOCABULARY = {
    # Core Christian Theology
    'salvation', 'redemption', 'justification', 'sanctification', 'glorification',
    'atonement', 'propitiation', 'reconciliation', 'forgiveness', 'mercy', 'grace',
    'faith', 'belief', 'trust', 'hope', 'love', 'charity', 'righteousness',
    'holiness', 'purity', 'obedience', 'repentance', 'confession', 'conversion',
    
    # God and Trinity
    'trinity', 'godhead', 'deity', 'divinity', 'sovereignty', 'omnipotence',
    'omniscience', 'omnipresence', 'immutability', 'eternality', 'creator',
    
    # Jesus Christ
    'messiah', 'christ', 'savior', 'lord', 'redeemer', 'mediator', 'advocate',
    'lamb', 'shepherd', 'king', 'prophet', 'priest', 'incarnation', 'virgin',
    'crucifixion', 'resurrection', 'ascension', 'intercession', 'return',
    
    # Holy Spirit
    'spirit', 'comforter', 'counselor', 'advocate', 'indwelling', 'filling',
    'anointing', 'empowerment', 'conviction', 'illumination',
    
    # Scripture and Revelation
    'inspiration', 'inerrancy', 'infallibility', 'authority', 'revelation',
    'prophecy', 'fulfillment', 'covenant', 'promise', 'commandment',
    
    # Sin and Fall
    'sin', 'transgression', 'iniquity', 'wickedness', 'depravity', 'corruption',
    'temptation', 'flesh', 'worldliness', 'idolatry', 'pride', 'rebellion',
    
    # Salvation Process
    'election', 'predestination', 'calling', 'regeneration', 'adoption',
    'imputation', 'perseverance', 'assurance', 'security', 'eternal',
    
    # Church and Community
    'church', 'congregation', 'fellowship', 'communion', 'baptism', 'ordinance',
    'sacrament', 'membership', 'discipline', 'unity', 'body', 'bride',
    
    # Christian Life
    'discipleship', 'stewardship', 'service', 'ministry', 'witness', 'testimony',
    'evangelism', 'missions', 'prayer', 'worship', 'praise', 'thanksgiving',
    'fasting', 'meditation', 'devotion', 'consecration', 'dedication',
    
    # Spiritual Warfare
    'warfare', 'armor', 'victory', 'deliverance', 'freedom', 'bondage',
    'stronghold', 'spiritual', 'demonic', 'satan', 'devil', 'angels',
    
    # End Times / Eschatology
    'rapture', 'tribulation', 'millennium', 'judgment', 'heaven', 'hell',
    'eternity', 'paradise', 'kingdom', 'throne', 'glory', 'reward',
    
    # Biblical Characters and Groups
    'apostle', 'disciple', 'prophet', 'patriarch', 'priest', 'levite',
    'pharisee', 'gentile', 'jew', 'israelite', 'hebrew', 'christian',
    
    # Biblical Events and Themes
    'creation', 'fall', 'flood', 'exodus', 'passover', 'wilderness',
    'promised', 'exile', 'restoration', 'birth', 'death', 'burial',
    'christmas', 'easter', 'pentecost', 'advent',
    
    # Virtues and Fruit of Spirit
    'patience', 'kindness', 'goodness', 'faithfulness', 'gentleness',
    'self-control', 'humility', 'meekness', 'temperance', 'contentment',
    
    # Biblical Locations (significant ones)
    'jerusalem', 'bethlehem', 'nazareth', 'galilee', 'calvary', 'golgotha',
    'gethsemane', 'jordan', 'egypt', 'babylon', 'israel', 'zion',
    
    # Family and Relationships
    'marriage', 'family', 'children', 'parenting', 'husband', 'wife',
    'father', 'mother', 'brother', 'sister', 'widow', 'orphan',
    
    # Money and Possessions
    'tithe', 'offering', 'giving', 'generosity', 'treasure', 'riches',
    'poverty', 'provision', 'blessing', 'prosperity',
    
    # Suffering and Trials
    'suffering', 'persecution', 'trial', 'tribulation', 'affliction',
    'comfort', 'healing', 'restoration', 'hope', 'endurance',
    
    # Leadership and Authority
    'pastor', 'elder', 'deacon', 'bishop', 'overseer', 'shepherd',
    'teacher', 'preacher', 'evangelist', 'missionary', 'servant',
    
    # Worship Elements
    'hymn', 'psalm', 'song', 'altar', 'sacrifice', 'tabernacle', 'temple',
    'ark', 'incense', 'oil', 'bread', 'wine', 'cup', 'cross', 'blood',
    
    # Magi/Christmas specific
    'magi', 'wisemen', 'star', 'gold', 'frankincense', 'myrrh', 'gifts',
    'herod', 'angels', 'shepherds', 'manger', 'nativity', 'immanuel',
}


def extract_tags(text: str, quote_boundaries: List[QuoteBoundary] = None,
                 min_occurrences: int = 3, min_score: float = 0.20,
                 max_tags: int = 20, verbose: bool = True) -> List[str]:
    """
    Extract religious keyword tags from a Christian/Baptist/Evangelical sermon transcript.
    
    Uses a curated vocabulary of religious terms and requires significant evidence
    (multiple occurrences) in the text. Only returns tags with strong textual support.
    
    Args:
        text: The transcript text (with or without paragraphs)
        quote_boundaries: Quote boundaries to exclude quoted Bible text from analysis
        min_occurrences: Minimum times a word must appear to be considered (default: 3)
        min_score: Minimum KeyBERT relevance score to include (default: 0.25)
        max_tags: Maximum number of tags to return (default: 20)
        verbose: Whether to print progress messages
    
    Returns:
        List of keyword strings (only those with significant evidence), or empty list
    """
    global tag_model
    
    if not KEYBERT_AVAILABLE:
        if verbose:
            print("   ⚠️  KeyBERT not installed")
            print("   Install with: pip install keybert")
        return []
    
    # Load the high-quality model on first use
    if tag_model is None:
        if verbose:
            print(f"   Loading high-quality model ({TAG_MODEL_NAME})...")
        tag_model = SentenceTransformer(TAG_MODEL_NAME)
        if device == "mps":
            tag_model = tag_model.to(device)
            if verbose:
                print(f"   ✓ Tag model loaded on GPU")
    
    if verbose:
        print("   Analyzing sermon for religious themes...")
    
    # Remove Bible quotes from the text to avoid extracting quoted scripture phrases
    clean_text = text.lower()
    if quote_boundaries:
        sorted_boundaries = sorted(quote_boundaries, key=lambda x: x.start_pos, reverse=True)
        for qb in sorted_boundaries:
            clean_text = clean_text[:qb.start_pos] + " " + clean_text[qb.end_pos:]
        if verbose:
            print(f"   Excluded {len(quote_boundaries)} Bible quotes from analysis")
    
    # Count occurrences of each religious term in the text
    word_counts = {}
    for term in RELIGIOUS_TAG_VOCABULARY:
        # Use word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(term) + r'\b'
        count = len(re.findall(pattern, clean_text, re.IGNORECASE))
        if count >= min_occurrences:
            word_counts[term] = count
    
    if not word_counts:
        if verbose:
            print("   No religious terms found with sufficient evidence")
        return []
    
    if verbose:
        print(f"   Found {len(word_counts)} terms with {min_occurrences}+ occurrences")
    
    try:
        # Use KeyBERT with the high-quality model to rank the candidate terms
        # by semantic relevance to the overall sermon content
        kw_model = KeyBERT(model=tag_model)
        
        # Get the candidates that passed the occurrence filter
        candidates = list(word_counts.keys())
        
        # Extract keywords using the candidates parameter
        # This tells KeyBERT to only consider our religious vocabulary
        keywords = kw_model.extract_keywords(
            clean_text,
            candidates=candidates,
            top_n=len(candidates),  # Get scores for all candidates
            use_mmr=True,
            diversity=0.5,  # Moderate diversity
        )
        
        # Filter by score and format results
        final_tags = []
        for keyword, score in keywords:
            if score >= min_score:
                # Capitalize for display
                tag = keyword.title()
                final_tags.append((tag, score, word_counts[keyword.lower()]))
        
        # Sort by a combination of score and frequency
        # Higher score and higher frequency = better tag
        final_tags.sort(key=lambda x: x[1] * (1 + x[2] / 10), reverse=True)
        
        # Take top tags up to max
        result_tags = [tag for tag, score, count in final_tags[:max_tags]]
        
        if verbose:
            print(f"   ✓ Extracted {len(result_tags)} religious tags")
            if result_tags and len(final_tags) > 0:
                # Show evidence for top tags
                for tag, score, count in final_tags[:min(3, len(final_tags))]:
                    print(f"      • {tag}: {count} occurrences, relevance {score:.2f}")
        
        return result_tags
        
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Error extracting tags: {str(e)}")
        return []


if __name__ == "__main__":
    import sys
    
    # Check for test mode flag
    test_mode = "test" in sys.argv
    
    # Get input file from command line or use default (skip "test" argument)
    args = [arg for arg in sys.argv[1:] if arg != "test"]
    if args:
        audio_file = args[0]
    else:
        audio_file = "20251214-SunAM-Polar.mp3"
    
    # PIPELINE ORDER (optimized):
    # 1. Transcribe audio to raw text (or load from test file)
    # 2. Process Bible quotes (auto-detect translation + normalize references + add quotation marks)
    # 3. Segment into paragraphs (respecting quote boundaries)
    # 4. Extract scripture references
    # 5. Extract keyword tags for categorization
    # 6. Convert to final markdown file (final.md)
    
    print("\n" + "=" * 70)
    print("SERMON TRANSCRIPTION PIPELINE")
    if test_mode:
        print("Mode: TEST (using whisper_test.txt)")
    else:
        print(f"Input: {audio_file}")
    print("Bible Translation: AUTO-DETECT (per-quote)")
    print("=" * 70)
    
    # Step 1: Transcribe audio OR load test file
    if test_mode:
        print("\n📝 STEP 1: Loading test file (whisper_test.txt)...")
        with open("whisper_test.txt", "r", encoding="utf-8") as f:
            raw = f.read()
        print("   ✓ Loaded test transcription from: whisper_test.txt")
    else:
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
    
    # Step 4: Build scripture references section
    print("\n📖 STEP 4: Building scripture references...")
    references_section = ""
    if quote_boundaries:
        # Extract unique references, preserving order of first appearance
        # Use the formatted reference string (e.g., "Matthew 2:1-12")
        seen_refs = set()
        unique_refs = []
        for qb in quote_boundaries:
            # Get the properly formatted reference string
            ref_str = qb.reference.to_standard_format()
            if ref_str not in seen_refs:
                seen_refs.add(ref_str)
                unique_refs.append(ref_str)
        
        if unique_refs:
            references_section = "\n\n---\n\n## Scripture References\n\n"
            references_section += "\n".join(f"- {ref}" for ref in unique_refs)
            print(f"   ✓ Found {len(unique_refs)} unique scripture references")
    else:
        print("   No scripture references found")
    
    # Step 5: Extract keyword tags for categorization
    print("\n🏷️  STEP 5: Extracting keyword tags...")
    tags = extract_tags(paragraphed, quote_boundaries=quote_boundaries, verbose=True)
    tags_section = ""
    if tags:
        tags_section = "\n\n---\n\n## Tags\n\n"
        tags_section += ", ".join(tags)
    
    # Append references and tags to the final output
    final_output = paragraphed
    if references_section:
        final_output += references_section
    if tags_section:
        final_output += tags_section
    
    # Save final output (plain text version)
    with open("whisper_cleaned.txt", "w", encoding="utf-8") as f:
        f.write(final_output)
    
    # Step 6: Convert to formatted markdown file
    print("\n📝 STEP 6: Converting to markdown (final.md)...")
    markdown_output = convert_to_markdown(
        transcript=paragraphed,
        quote_boundaries=quote_boundaries,
        tags=tags,
        scripture_refs=unique_refs if quote_boundaries else None
    )
    
    # Save markdown output
    with open("final.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print("   ✓ Markdown file saved to: final.md")
    
    print("\n" + "=" * 70)
    print("✅ TRANSCRIPTION COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    if not test_mode:
        print("  • whisper_raw.txt      - Raw transcription (no processing)")
    print("  • whisper_quotes.txt   - With Bible quotes marked")
    print("  • whisper_cleaned.txt  - Final output with paragraphs")
    print("  • final.md             - Formatted markdown (tags, refs, italics, bold)")
    print(f"\nPipeline:")
    if test_mode:
        print("  1. ✓ Test file loaded (whisper_test.txt)")
    else:
        print("  1. ✓ Audio transcription (Whisper medium model)")
    print("  2. ✓ Bible translation auto-detection (per-quote)")
    print("  3. ✓ Bible quote detection and normalization")
    print("  4. ✓ Paragraph segmentation (quote-aware)")
    print("  5. ✓ Scripture references extracted")
    if tags:
        print(f"  6. ✓ Keyword tags extracted ({len(tags)} tags)")
    else:
        print("  6. ⚠️  Tag extraction skipped (KeyBERT not available)")
    print("  7. ✓ Markdown conversion (final.md)")
    print("=" * 70)