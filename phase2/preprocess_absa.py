"""
=============================================================================
Aspect-Based Sentiment Analysis (ABSA) Dataset Preprocessing
=============================================================================

This script preprocesses the SEntFiN dataset for entity-aware sentiment analysis.
For headlines with multiple entities, it creates multiple training examples where:
- The target entity is replaced with [TARGET]
- Other entities are replaced with [OTHER]
- The sentiment label corresponds to the target entity

Example:
  Input: "Reliance profits soar while Tata Steel shares plummet"
  Decisions: {'Reliance': 'positive', 'Tata Steel': 'negative'}
  
  Output examples:
    1. "[TARGET] profits soar while [OTHER] shares plummet" → positive
    2. "[OTHER] profits soar while [TARGET] shares plummet" → negative

Author: NLP Engineering Team
Date: 2024
=============================================================================
"""

import pandas as pd
import ast
import re
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENTITY MAPPING LOADING
# =============================================================================

def load_entity_mappings(entity_file: str) -> Dict[str, str]:
    """
    Load entity mappings from entity_list_comprehensive.csv.
    
    Creates a mapping from all entity name variations to their canonical symbol.
    This helps match entities in text even if they appear in different forms.
    
    Args:
        entity_file: Path to entity_list_comprehensive.csv
        
    Returns:
        Dictionary mapping entity name variations to their symbols
        Example: {'Reliance Industries': 'RELIANCE', 'Reliance Ind': 'RELIANCE', ...}
    """
    logger.info("=" * 80)
    logger.info("LOADING ENTITY MAPPINGS")
    logger.info("=" * 80)
    
    try:
        df = pd.read_csv(entity_file)
        logger.info(f"✓ Loaded {len(df)} entity entries")
    except FileNotFoundError:
        logger.warning(f"Entity file {entity_file} not found. Continuing without entity mapping.")
        return {}
    
    entity_map = {}
    
    for _, row in df.iterrows():
        symbol = row.get('Symbol', '')
        entity_str = row.get('Entity', '')
        
        if pd.isna(symbol) or pd.isna(entity_str):
            continue
        
        # Parse entity variations (comma-separated)
        # Example: "Reliance Industries Ltd., Reliance Industries, Reliance Ind, RELIANCE"
        variations = [v.strip() for v in str(entity_str).split(',')]
        
        # Map each variation to the symbol
        for variation in variations:
            if variation:
                # Store both original case and lowercase for matching
                entity_map[variation] = symbol
                entity_map[variation.lower()] = symbol
    
    logger.info(f"✓ Created {len(entity_map)} entity name mappings")
    return entity_map


def build_entity_patterns(entity_map: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Build regex patterns for entity matching.
    
    Returns list of (pattern, symbol, original_name) tuples sorted by length
    (longest first) to match longer entity names before shorter ones.
    
    Args:
        entity_map: Dictionary mapping entity names to symbols
        
    Returns:
        List of (pattern, symbol, original_name) tuples
    """
    patterns = []
    
    # Group by symbol to avoid duplicate patterns
    symbol_to_names = defaultdict(set)
    for name, symbol in entity_map.items():
        symbol_to_names[symbol].add(name)
    
    for symbol, names in symbol_to_names.items():
        for name in names:
            # Escape special regex characters
            pattern = re.escape(name)
            # Use word boundaries for exact matching
            pattern = r'\b' + pattern + r'\b'
            patterns.append((pattern, symbol, name))
    
    # Sort by length (longest first) for better matching
    patterns.sort(key=lambda x: len(x[2]), reverse=True)
    
    return patterns


# =============================================================================
# DECISIONS PARSING
# =============================================================================

def parse_decisions(decisions_str: str) -> Optional[Dict[str, str]]:
    """
    Parse the Decisions column string as Python dictionary.
    
    Args:
        decisions_str: String representation of Python dictionary
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    if pd.isna(decisions_str) or decisions_str == '':
        return None
    
    try:
        decisions_str = decisions_str.strip()
        # Remove outer quotes if present
        if decisions_str.startswith('"') and decisions_str.endswith('"'):
            decisions_str = decisions_str[1:-1]
        if decisions_str.startswith("'") and decisions_str.endswith("'") and len(decisions_str) > 2:
            if not decisions_str.startswith("{'") and not decisions_str.startswith("{"):
                decisions_str = decisions_str[1:-1]
        
        # Parse Python dictionary string
        decisions_dict = ast.literal_eval(decisions_str)
        
        if not isinstance(decisions_dict, dict):
            return None
            
        return decisions_dict
    except (ValueError, SyntaxError):
        return None


# =============================================================================
# ENTITY MATCHING IN TEXT
# =============================================================================

def find_entities_in_text(text: str, entity_patterns: List[Tuple[str, str, str]]) -> List[Tuple[str, int, int]]:
    """
    Find all entity mentions in text.
    
    Args:
        text: Input text
        entity_patterns: List of (pattern, symbol, original_name) tuples
        
    Returns:
        List of (entity_name, start_pos, end_pos) tuples, sorted by position
    """
    matches = []
    text_lower = text.lower()
    
    for pattern, symbol, original_name in entity_patterns:
        # Try case-insensitive matching
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span()
            # Avoid overlapping matches (prefer longer matches)
            overlaps = False
            for existing_name, existing_start, existing_end in matches:
                if not (end <= existing_start or start >= existing_end):
                    overlaps = True
                    break
            if not overlaps:
                matches.append((original_name, start, end))
    
    # Sort by position
    matches.sort(key=lambda x: x[1])
    return matches


def match_entities_to_decisions(
    text: str,
    decisions_dict: Dict[str, str],
    entity_patterns: List[Tuple[str, str, str]]
) -> Dict[str, str]:
    """
    Match entities from Decisions dictionary to their mentions in text.
    
    Args:
        text: Input text
        decisions_dict: Dictionary from Decisions column {entity: sentiment}
        entity_patterns: List of (pattern, symbol, original_name) tuples
        
    Returns:
        Dictionary mapping entity names found in text to their sentiments
    """
    found_entities = {}
    text_lower = text.lower()
    
    # First, try exact matching from decisions_dict
    for entity_name, sentiment in decisions_dict.items():
        # Try various matching strategies
        entity_lower = entity_name.lower()
        
        # Exact match (case-insensitive)
        if entity_lower in text_lower:
            found_entities[entity_name] = sentiment
            continue
        
        # Word boundary match
        pattern = r'\b' + re.escape(entity_name) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_entities[entity_name] = sentiment
            continue
        
        # Partial match (entity name contains key words)
        entity_words = set(entity_lower.split())
        text_words = set(text_lower.split())
        if len(entity_words) > 0 and entity_words.issubset(text_words):
            # Check if it's a reasonable match (not too generic)
            if len(entity_words) >= 2 or len(entity_name) >= 5:
                found_entities[entity_name] = sentiment
    
    return found_entities


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean text using same approach as Phase 1, but preserve [TARGET] and [OTHER] placeholders.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    
    # Temporarily replace placeholders with unique tokens that won't be affected by cleaning
    # Use tokens that are valid after cleaning (lowercase letters and numbers)
    text = text.replace('[TARGET]', 'TARGETPLACEHOLDER')
    text = text.replace('[OTHER]', 'OTHERPLACEHOLDER')
    
    # Standard cleaning
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Restore placeholders (in uppercase as specified)
    text = text.replace('targetplaceholder', '[TARGET]')
    text = text.replace('otherplaceholder', '[OTHER]')
    
    return text


def replace_entities_with_placeholders(
    text: str,
    target_entity: str,
    other_entities: List[str]
) -> str:
    """
    Replace entities in text with [TARGET] and [OTHER] placeholders.
    
    Args:
        text: Original text
        target_entity: Entity to replace with [TARGET]
        other_entities: List of entities to replace with [OTHER]
        
    Returns:
        Text with entities replaced by placeholders
    """
    result = text
    
    # Replace other entities first (to avoid conflicts)
    for entity in other_entities:
        # Use word boundaries for exact matching
        pattern = r'\b' + re.escape(entity) + r'\b'
        result = re.sub(pattern, '[OTHER]', result, flags=re.IGNORECASE)
    
    # Replace target entity
    pattern = r'\b' + re.escape(target_entity) + r'\b'
    result = re.sub(pattern, '[TARGET]', result, flags=re.IGNORECASE)
    
    return result


def map_sentiment(sentiment: str) -> int:
    """
    Map sentiment string to numeric value.
    
    Args:
        sentiment: Sentiment string ('positive', 'negative', 'neutral')
        
    Returns:
        Numeric sentiment: 1 (positive), -1 (negative), 0 (neutral)
    """
    sentiment_lower = sentiment.lower().strip()
    sentiment_mapping = {
        'positive': 1,
        'negative': -1,
        'neutral': 0
    }
    return sentiment_mapping.get(sentiment_lower, 0)


# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================

def preprocess_absa_dataset(
    input_file: str,
    entity_file: str,
    output_file: str
) -> pd.DataFrame:
    """
    Main preprocessing function for ABSA dataset.
    
    Args:
        input_file: Path to SEntFiN.csv
        entity_file: Path to entity_list_comprehensive.csv
        output_file: Path to output absa_dataset.csv
        
    Returns:
        Processed DataFrame
    """
    logger.info("=" * 80)
    logger.info("ABSA DATASET PREPROCESSING")
    logger.info("=" * 80)
    
    # Load entity mappings
    entity_map = load_entity_mappings(entity_file)
    entity_patterns = build_entity_patterns(entity_map)
    
    # Load dataset
    logger.info(f"\nLoading dataset from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        logger.info(f"✓ Loaded {len(df)} rows")
    except FileNotFoundError:
        logger.error(f"Error: File {input_file} not found!")
        raise
    
    # Initialize output lists
    processed_data = []
    
    # Statistics
    stats = {
        'total_rows': len(df),
        'single_entity': 0,
        'multi_entity': 0,
        'parse_errors': 0,
        'no_entities_found': 0,
        'examples_created': 0,
        'sentiment_dist': defaultdict(int)
    }
    
    # Process each row
    logger.info("\nProcessing rows...")
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} rows...")
        
        # Get title
        title = row.get('Title', '')
        if pd.isna(title):
            title = ''
        title = str(title).strip()
        
        if not title:
            continue
        
        # Parse Decisions
        decisions_str = row.get('Decisions', '')
        decisions_dict = parse_decisions(decisions_str)
        
        if decisions_dict is None:
            stats['parse_errors'] += 1
            continue
        
        num_entities = len(decisions_dict)
        
        if num_entities == 0:
            continue
        elif num_entities == 1:
            # Single entity: create one example
            stats['single_entity'] += 1
            entity_name = list(decisions_dict.keys())[0]
            sentiment = decisions_dict[entity_name]
            
            # Replace entity with [TARGET]
            processed_text = replace_entities_with_placeholders(
                title, entity_name, []
            )
            
            # Clean text
            cleaned_text = clean_text(processed_text)
            
            if cleaned_text:
                processed_data.append({
                    'headline': title,
                    'clean_text': cleaned_text,
                    'target_entity': entity_name,
                    'sentiment': map_sentiment(sentiment),
                    'sentiment_str': sentiment
                })
                stats['examples_created'] += 1
                stats['sentiment_dist'][sentiment.lower()] += 1
        else:
            # Multiple entities: create one example per entity
            stats['multi_entity'] += 1
            
            # Match entities to text
            found_entities = match_entities_to_decisions(
                title, decisions_dict, entity_patterns
            )
            
            if not found_entities:
                stats['no_entities_found'] += 1
                # Fallback: use all entities from decisions
                found_entities = decisions_dict
            
            # Create example for each entity
            entity_list = list(found_entities.keys())
            for target_entity in entity_list:
                sentiment = found_entities[target_entity]
                other_entities = [e for e in entity_list if e != target_entity]
                
                # Replace entities with placeholders
                processed_text = replace_entities_with_placeholders(
                    title, target_entity, other_entities
                )
                
                # Clean text
                cleaned_text = clean_text(processed_text)
                
                if cleaned_text:
                    processed_data.append({
                        'headline': title,
                        'clean_text': cleaned_text,
                        'target_entity': target_entity,
                        'sentiment': map_sentiment(sentiment),
                        'sentiment_str': sentiment
                    })
                    stats['examples_created'] += 1
                    stats['sentiment_dist'][sentiment.lower()] += 1
    
    # Create output DataFrame
    logger.info("\n" + "=" * 80)
    logger.info("CREATING OUTPUT DATASET")
    logger.info("=" * 80)
    
    processed_df = pd.DataFrame(processed_data)
    
    # Save to CSV
    logger.info(f"\nSaving processed dataset to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved {len(processed_df)} examples to {output_file}")
    
    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total input rows: {stats['total_rows']:,}")
    logger.info(f"Single-entity headlines: {stats['single_entity']:,}")
    logger.info(f"Multi-entity headlines: {stats['multi_entity']:,}")
    logger.info(f"Parse errors: {stats['parse_errors']:,}")
    logger.info(f"No entities found in text: {stats['no_entities_found']:,}")
    logger.info(f"Total examples created: {stats['examples_created']:,}")
    logger.info(f"\nSentiment distribution:")
    for sentiment, count in sorted(stats['sentiment_dist'].items()):
        logger.info(f"  {sentiment}: {count:,}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    
    return processed_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # File paths
    input_file = 'SEntFiN.csv'
    entity_file = 'entity_list_comprehensive.csv'
    output_file = 'absa_dataset.csv'
    
    # Run preprocessing
    processed_df = preprocess_absa_dataset(
        input_file=input_file,
        entity_file=entity_file,
        output_file=output_file
    )
    
    # Display sample
    logger.info("\nSample of processed data:")
    logger.info(processed_df.head(10).to_string())

