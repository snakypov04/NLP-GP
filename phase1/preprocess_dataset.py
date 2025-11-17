"""
Financial Sentiment Analysis Dataset Preprocessing Script

This script preprocesses the SEntFiN dataset to create a clean single-entity
sentiment analysis dataset by:
1. Filtering out multi-entity headlines
2. Extracting single sentiment values
3. Cleaning text (lowercase, remove URLs, special chars, etc.)
4. Mapping sentiment labels to numeric values
"""

import pandas as pd
import json
import ast
import re
from typing import Dict, Optional, Tuple
import sys


def parse_decisions(decisions_str: str) -> Optional[Dict[str, str]]:
    """
    Parse the Decisions column string as Python dictionary.
    The Decisions column contains Python dictionary strings (with single quotes),
    not JSON, so we use ast.literal_eval() instead of json.loads().
    
    Args:
        decisions_str: String representation of Python dictionary
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    if pd.isna(decisions_str) or decisions_str == '':
        return None
    
    try:
        # Handle cases where the string might have extra quotes
        decisions_str = decisions_str.strip()
        # Remove outer quotes if present (CSV might have double-quoted strings)
        if decisions_str.startswith('"') and decisions_str.endswith('"'):
            decisions_str = decisions_str[1:-1]
        if decisions_str.startswith("'") and decisions_str.endswith("'") and len(decisions_str) > 2:
            # Only remove if it's a quoted string, not a dict starting with single quote
            # Check if it's actually a quoted string vs dict syntax
            if not decisions_str.startswith("{'") and not decisions_str.startswith("{"):
                decisions_str = decisions_str[1:-1]
        
        # Parse Python dictionary string using ast.literal_eval (safer than eval)
        decisions_dict = ast.literal_eval(decisions_str)
        
        # Ensure it's a dictionary
        if not isinstance(decisions_dict, dict):
            return None
            
        return decisions_dict
    except (ValueError, SyntaxError) as e:
        # Silently skip parse errors (too many to print)
        return None


def count_entities(decisions_dict: Optional[Dict[str, str]]) -> int:
    """
    Count the number of entities in the Decisions dictionary.
    
    Args:
        decisions_dict: Dictionary with entity-sentiment pairs
        
    Returns:
        Number of entities (0 if None or empty)
    """
    if decisions_dict is None:
        return 0
    return len(decisions_dict)


def extract_single_sentiment(decisions_dict: Optional[Dict[str, str]]) -> Optional[str]:
    """
    Extract sentiment value from single-entity Decisions dictionary.
    
    Args:
        decisions_dict: Dictionary with exactly one entity-sentiment pair
        
    Returns:
        Sentiment value ('positive', 'neutral', 'negative') or None
    """
    if decisions_dict is None or len(decisions_dict) != 1:
        return None
    
    # Get the first (and only) sentiment value
    sentiment = list(decisions_dict.values())[0]
    return sentiment


def clean_text(text: str) -> str:
    """
    Clean text by:
    - Converting to lowercase
    - Removing URLs (http/https/www)
    - Removing special characters except apostrophes
    - Removing extra whitespace
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs (http://, https://, www.)
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    
    # Remove special characters except apostrophes and spaces
    # Keep: letters, numbers, apostrophes, spaces
    text = re.sub(r"[^a-z0-9'\s]", '', text)
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def map_sentiment(sentiment: str) -> int:
    """
    Map sentiment label to numeric value.
    
    Args:
        sentiment: Sentiment label ('positive', 'neutral', 'negative')
        
    Returns:
        Numeric sentiment value (1, 0, -1)
    """
    sentiment_mapping = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }
    
    sentiment_lower = sentiment.lower().strip()
    return sentiment_mapping.get(sentiment_lower, 0)  # Default to 0 (neutral) if unknown


def preprocess_dataset(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Main preprocessing function to filter and clean the dataset.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        
    Returns:
        Processed DataFrame
    """
    print(f"Loading dataset from {input_file}...")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    print(f"Total rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize lists for processed data
    headlines = []
    clean_texts = []
    sentiments = []
    
    # Track statistics
    total_rows = len(df)
    single_entity_count = 0
    multi_entity_count = 0
    parse_error_count = 0
    missing_decisions_count = 0
    sentiment_distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
    
    # Process each row
    print("\nProcessing rows...")
    for idx, row in df.iterrows():
        # Get title (headline)
        title = row.get('Title', '')
        if pd.isna(title):
            title = ''
        
        # Parse Decisions column
        decisions_str = row.get('Decisions', '')
        decisions_dict = parse_decisions(decisions_str)
        
        # Handle parsing errors
        if decisions_dict is None:
            if pd.isna(decisions_str) or decisions_str == '':
                missing_decisions_count += 1
            else:
                parse_error_count += 1
            continue
        
        # Count entities
        entity_count = count_entities(decisions_dict)
        
        # Filter: only keep single-entity rows
        if entity_count != 1:
            multi_entity_count += 1
            continue
        
        # Extract sentiment
        sentiment_str = extract_single_sentiment(decisions_dict)
        if sentiment_str is None:
            continue
        
        # Clean text
        cleaned = clean_text(title)
        
        # Skip if cleaned text is empty
        if cleaned == '':
            continue
        
        # Map sentiment to numeric value
        sentiment_numeric = map_sentiment(sentiment_str)
        
        # Store processed data
        headlines.append(title)
        clean_texts.append(cleaned)
        sentiments.append(sentiment_numeric)
        
        # Update statistics
        single_entity_count += 1
        sentiment_distribution[sentiment_str.lower()] = sentiment_distribution.get(
            sentiment_str.lower(), 0
        ) + 1
    
    # Create output DataFrame
    processed_df = pd.DataFrame({
        'headline': headlines,
        'clean_text': clean_texts,
        'sentiment': sentiments
    })
    
    # Save to CSV
    print(f"\nSaving processed dataset to {output_file}...")
    processed_df.to_csv(output_file, index=False)
    print(f"Saved {len(processed_df)} rows to {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("PREPROCESSING STATISTICS")
    print("="*60)
    print(f"Total headlines in input: {total_rows}")
    print(f"Single-entity headlines: {single_entity_count}")
    print(f"Multi-entity headlines (filtered out): {multi_entity_count}")
    print(f"Parse errors: {parse_error_count}")
    print(f"Missing Decisions: {missing_decisions_count}")
    print(f"\nFinal dataset size: {len(processed_df)} rows")
    print(f"\nSentiment Distribution:")
    print(f"  Positive (1): {sentiment_distribution.get('positive', 0)}")
    print(f"  Neutral (0): {sentiment_distribution.get('neutral', 0)}")
    print(f"  Negative (-1): {sentiment_distribution.get('negative', 0)}")
    print("="*60)
    
    return processed_df


if __name__ == "__main__":
    # File paths
    input_file = "SEntFiN.csv"
    output_file = "phase1_dataset.csv"
    
    # Run preprocessing
    processed_df = preprocess_dataset(input_file, output_file)
    
    # Display sample of processed data
    print("\nSample of processed data (first 5 rows):")
    print(processed_df.head())
    
    print("\nPreprocessing completed successfully!")

