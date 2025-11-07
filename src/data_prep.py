"""
Data preparation and cleaning functions for NYC Airbnb analysis.

This module contains functions for loading, cleaning, and preprocessing
the Airbnb dataset including handling missing values, outliers, and
feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import json
from datetime import datetime


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.
    
    Args:
        path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning operations on the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Standardize column names (lowercase, replace spaces with underscores)
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Trim whitespace from string columns
    string_cols = df_clean.select_dtypes(include=['object']).columns
    for col in string_cols:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Convert date columns
    if 'last_review' in df_clean.columns:
        df_clean['last_review'] = pd.to_datetime(df_clean['last_review'], errors='coerce')
    
    print("Basic cleaning completed:")
    print(f"- Standardized {len(df_clean.columns)} column names")
    print(f"- Trimmed whitespace from {len(string_cols)} string columns")
    print(f"- Converted date columns")
    
    return df_clean


def handle_missing(df: pd.DataFrame, strategy: Dict = None) -> pd.DataFrame:
    """
    Handle missing values based on specified strategy.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (Dict): Strategy for handling missing values
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    if strategy is None:
        strategy = {
            'reviews_per_month': 0,  # Fill with 0 (no reviews)
            'last_review': 'drop_if_no_reviews',  # Keep consistent with reviews_per_month
            'host_name': 'Unknown Host',
            'name': 'No Name Provided'
        }
    
    df_clean = df.copy()
    initial_shape = df_clean.shape
    
    # Handle reviews_per_month
    if 'reviews_per_month' in df_clean.columns:
        missing_reviews = df_clean['reviews_per_month'].isnull().sum()
        df_clean['reviews_per_month'].fillna(strategy['reviews_per_month'], inplace=True)
        print(f"Filled {missing_reviews} missing reviews_per_month with {strategy['reviews_per_month']}")
    
    # Handle host_name
    if 'host_name' in df_clean.columns:
        missing_hosts = df_clean['host_name'].isnull().sum()
        df_clean['host_name'].fillna(strategy['host_name'], inplace=True)
        print(f"Filled {missing_hosts} missing host_name with '{strategy['host_name']}'")
    
    # Handle name
    if 'name' in df_clean.columns:
        missing_names = df_clean['name'].isnull().sum()
        df_clean['name'].fillna(strategy['name'], inplace=True)
        print(f"Filled {missing_names} missing name with '{strategy['name']}'")
    
    # Drop rows missing essential fields
    essential_fields = ['id', 'host_id', 'neighbourhood_group', 'room_type', 'price']
    before_drop = len(df_clean)
    df_clean = df_clean.dropna(subset=essential_fields)
    after_drop = len(df_clean)
    dropped_essential = before_drop - after_drop
    
    if dropped_essential > 0:
        print(f"Dropped {dropped_essential} rows missing essential fields")
    
    print(f"Shape after handling missing values: {df_clean.shape}")
    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with duplicates removed
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=['id'], keep='first')
    final_count = len(df_clean)
    duplicates_removed = initial_count - final_count
    
    print(f"Removed {duplicates_removed} duplicate listings based on ID")
    return df_clean


def filter_price_outliers(df: pd.DataFrame, method: str = 'IQR', factor: float = 1.5) -> pd.DataFrame:
    """
    Remove price outliers using specified method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): Method for outlier detection ('IQR' or 'percentile')
        factor (float): Factor for IQR method
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    if method == 'IQR':
        Q1 = df_clean['price'].quantile(0.25)
        Q3 = df_clean['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Also remove zero prices
        df_clean = df_clean[(df_clean['price'] > 0) & 
                           (df_clean['price'] >= lower_bound) & 
                           (df_clean['price'] <= upper_bound)]
    
    elif method == 'percentile':
        # Remove bottom 1% and top 1% of prices, and zero prices
        lower_percentile = df_clean['price'].quantile(0.01)
        upper_percentile = df_clean['price'].quantile(0.99)
        df_clean = df_clean[(df_clean['price'] > 0) & 
                           (df_clean['price'] >= lower_percentile) & 
                           (df_clean['price'] <= upper_percentile)]
    
    final_count = len(df_clean)
    outliers_removed = initial_count - final_count
    
    print(f"Removed {outliers_removed} price outliers using {method} method")
    print(f"Price range after filtering: ${df_clean['price'].min():.0f} - ${df_clean['price'].max():.0f}")
    
    return df_clean


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    df_clean = df.copy()
    
    # Price per minimum night
    df_clean['price_per_minimum_night'] = df_clean['price'] * df_clean['minimum_nights']
    
    # Active days (inverse of availability - how booked the listing is)
    df_clean['active_days'] = 365 - df_clean['availability_365']
    
    # Last review age (days since last review)
    if 'last_review' in df_clean.columns:
        current_date = pd.Timestamp('2019-12-31')  # Assuming data is from 2019
        df_clean['last_review_age'] = (current_date - df_clean['last_review']).dt.days
        df_clean['last_review_age'].fillna(9999, inplace=True)  # Large number for never reviewed
    
    # Reviews per year (annualized)
    df_clean['reviews_per_year'] = df_clean['reviews_per_month'] * 12
    
    # Host productivity (listings per host)
    df_clean['host_productivity'] = df_clean['calculated_host_listings_count']
    
    print("Feature engineering completed:")
    print("- price_per_minimum_night: Total cost for minimum stay")
    print("- active_days: Days booked per year (365 - availability)")
    print("- last_review_age: Days since last review")
    print("- reviews_per_year: Annualized review rate")
    print("- host_productivity: Number of listings per host")
    
    return df_clean


def save_missing_summary(df_before: pd.DataFrame, df_after: pd.DataFrame, output_path: str) -> None:
    """
    Save missing value summary before and after cleaning.
    
    Args:
        df_before (pd.DataFrame): Dataset before cleaning
        df_after (pd.DataFrame): Dataset after cleaning
        output_path (str): Path to save JSON summary
    """
    summary = {
        'before_cleaning': {
            'total_rows': len(df_before),
            'missing_values': df_before.isnull().sum().to_dict()
        },
        'after_cleaning': {
            'total_rows': len(df_after),
            'missing_values': df_after.isnull().sum().to_dict()
        },
        'cleaning_summary': {
            'rows_removed': len(df_before) - len(df_after),
            'removal_percentage': ((len(df_before) - len(df_after)) / len(df_before)) * 100
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Missing value summary saved to {output_path}")


def run_data_cleaning_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Run the complete data cleaning pipeline.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save cleaned CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("=== STARTING DATA CLEANING PIPELINE ===")
    
    # Load data
    df = load_data(input_path)
    df_original = df.copy()
    
    # Apply cleaning steps
    df = basic_clean(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = filter_price_outliers(df, method='none')  # Keep all listings, only remove zero prices
    df = feature_engineer(df)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to {output_path}")
    
    # Save missing value summary
    missing_summary_path = output_path.replace('.csv', '_missing_summary.json')
    save_missing_summary(df_original, df, missing_summary_path)
    
    # Final summary
    print(f"\n=== CLEANING PIPELINE COMPLETE ===")
    print(f"Original shape: {df_original.shape}")
    print(f"Final shape: {df.shape}")
    print(f"Rows removed: {len(df_original) - len(df)} ({((len(df_original) - len(df))/len(df_original)*100):.1f}%)")
    
    return df


# Sanity check functions
def validate_cleaned_data(df: pd.DataFrame) -> bool:
    """
    Perform sanity checks on cleaned data.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        bool: True if all checks pass
    """
    checks_passed = 0
    total_checks = 0
    
    # Check 1: No duplicate IDs
    total_checks += 1
    if df['id'].nunique() == len(df):
        print("✓ No duplicate IDs found")
        checks_passed += 1
    else:
        print("✗ Duplicate IDs found")
    
    # Check 2: Price >= 0
    total_checks += 1
    if (df['price'] > 0).all():
        print("✓ All prices are positive")
        checks_passed += 1
    else:
        print("✗ Some prices are zero or negative")
    
    # Check 3: Valid coordinates
    total_checks += 1
    valid_coords = ((df['latitude'].between(40.4, 41.0)) & 
                   (df['longitude'].between(-74.3, -73.7))).all()
    if valid_coords:
        print("✓ All coordinates are within NYC bounds")
        checks_passed += 1
    else:
        print("✗ Some coordinates are outside NYC bounds")
    
    # Check 4: Valid room types
    total_checks += 1
    valid_room_types = {'Entire home/apt', 'Private room', 'Shared room'}
    if set(df['room_type'].unique()).issubset(valid_room_types):
        print("✓ All room types are valid")
        checks_passed += 1
    else:
        print("✗ Invalid room types found")
    
    print(f"\nSanity checks: {checks_passed}/{total_checks} passed")
    return checks_passed == total_checks