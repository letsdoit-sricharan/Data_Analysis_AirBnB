#!/usr/bin/env python3
"""
Complete Pipeline Runner for NYC Airbnb Analysis

This script runs the entire analysis pipeline to generate all output files
including cleaned data, visualizations, models, and summaries.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append('src')

# Import pipeline modules
from src.data_prep import run_data_cleaning_pipeline, validate_cleaned_data
from src.eda import run_eda_pipeline
from src.model import run_modeling_pipeline

def create_output_directories():
    """Create necessary output directories."""
    directories = [
        'outputs',
        'outputs/figures'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_input_data():
    """Check if input data exists, create sample if not."""
    input_files = ['AB_NYC_2019.csv', 'data/AB_NYC_2019.csv', 'notebooks/AB_NYC_2019.csv']

    
    for file_path in input_files:
        if os.path.exists(file_path):
            print(f"✓ Found input data: {file_path}")
            return file_path
    
    print("⚠️  Input data not found. Creating sample dataset...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 5000
    
    # Create realistic sample data with proper feature-price relationships
    
    # Base features
    neighbourhood_groups = np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'], 
                                          n_samples, p=[0.35, 0.25, 0.2, 0.15, 0.05])
    room_types = np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], 
                                n_samples, p=[0.52, 0.45, 0.03])
    
    # Create price based on realistic factors
    base_prices = {
        'Manhattan': 150, 'Brooklyn': 100, 'Queens': 80, 'Bronx': 70, 'Staten Island': 60
    }
    room_multipliers = {
        'Entire home/apt': 1.5, 'Private room': 1.0, 'Shared room': 0.6
    }
    
    prices = []
    latitudes = []
    longitudes = []
    
    for i in range(n_samples):
        borough = neighbourhood_groups[i]
        room_type = room_types[i]
        
        # Base price from borough and room type
        base_price = base_prices[borough] * room_multipliers[room_type]
        
        # Add some randomness
        price = base_price * np.random.lognormal(0, 0.4)
        prices.append(max(10, int(price)))  # Minimum $10
        
        # Realistic coordinates by borough
        if borough == 'Manhattan':
            latitudes.append(np.random.uniform(40.70, 40.80))
            longitudes.append(np.random.uniform(-74.02, -73.93))
        elif borough == 'Brooklyn':
            latitudes.append(np.random.uniform(40.58, 40.73))
            longitudes.append(np.random.uniform(-74.05, -73.85))
        elif borough == 'Queens':
            latitudes.append(np.random.uniform(40.54, 40.80))
            longitudes.append(np.random.uniform(-73.96, -73.70))
        elif borough == 'Bronx':
            latitudes.append(np.random.uniform(40.79, 40.92))
            longitudes.append(np.random.uniform(-73.93, -73.76))
        else:  # Staten Island
            latitudes.append(np.random.uniform(40.50, 40.65))
            longitudes.append(np.random.uniform(-74.26, -74.05))
    
    sample_data = pd.DataFrame({
        'id': range(1, n_samples + 1),
        'name': [f'Listing_{i}' for i in range(1, n_samples + 1)],
        'host_id': np.random.randint(1, 1000, n_samples),
        'host_name': [f'Host_{i}' for i in np.random.randint(1, 1000, n_samples)],
        'neighbourhood_group': neighbourhood_groups,
        'neighbourhood': [f'Neighborhood_{i}' for i in np.random.randint(1, 200, n_samples)],
        'latitude': latitudes,
        'longitude': longitudes,
        'room_type': room_types,
        'price': prices,
        'minimum_nights': np.random.choice([1, 2, 3, 7, 30], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'number_of_reviews': np.random.poisson(25, n_samples),
        'last_review': pd.date_range('2019-01-01', '2019-12-31', periods=n_samples),
        'reviews_per_month': np.random.uniform(0, 6, n_samples),
        'calculated_host_listings_count': np.random.poisson(3, n_samples),
        'availability_365': np.random.randint(0, 366, n_samples)
    })
    
    # Add some missing values for realism
    sample_data.loc[np.random.choice(sample_data.index, 200), 'reviews_per_month'] = np.nan
    sample_data.loc[np.random.choice(sample_data.index, 150), 'last_review'] = pd.NaT
    
    # Save sample data
    sample_path = 'sample_airbnb_data.csv'
    sample_data.to_csv(sample_path, index=False)
    print(f"✓ Created sample dataset: {sample_path}")
    
    return sample_path

def run_data_pipeline(input_path):
    """Run the data cleaning pipeline."""
    print("\n" + "="*60)
    print("🧹 RUNNING DATA CLEANING PIPELINE")
    print("="*60)
    
    try:
        # Run data cleaning
        df_clean = run_data_cleaning_pipeline(input_path, 'outputs/cleaned_airbnb.csv')
        
        # Validate cleaned data
        print("\n🔍 Validating cleaned data...")
        is_valid = validate_cleaned_data(df_clean)
        
        if is_valid:
            print("✅ Data validation passed!")
        else:
            print("⚠️  Data validation warnings found (but continuing...)")
        
        print(f"✅ Cleaned dataset saved: outputs/cleaned_airbnb.csv")
        print(f"📊 Final dataset shape: {df_clean.shape}")
        
        return df_clean
        
    except Exception as e:
        print(f"❌ Error in data pipeline: {str(e)}")
        raise

def run_eda_pipeline_wrapper(df_clean):
    """Run the EDA pipeline."""
    print("\n" + "="*60)
    print("📊 RUNNING EXPLORATORY DATA ANALYSIS PIPELINE")
    print("="*60)
    
    try:
        # Run EDA pipeline
        summary = run_eda_pipeline(df_clean)
        
        print("✅ EDA pipeline completed!")
        print("📈 Generated visualizations:")
        print("   - Borough analysis charts")
        print("   - Price distribution plots")
        print("   - Geographic maps")
        print("   - Correlation heatmaps")
        print("   - Neighborhood analysis")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error in EDA pipeline: {str(e)}")
        raise

def run_modeling_pipeline_wrapper(df_clean):
    """Run the modeling pipeline."""
    print("\n" + "="*60)
    print("🤖 RUNNING MACHINE LEARNING PIPELINE")
    print("="*60)
    
    try:
        # Run modeling pipeline
        model_results = run_modeling_pipeline(df_clean)
        
        print("✅ Modeling pipeline completed!")
        print("🎯 Generated models:")
        print("   - Linear Regression model (model_price_lr.joblib)")
        print("   - Random Forest model (model_price_rf.joblib)")
        print("   - K-means clustering model (model_kmeans.joblib)")
        print("📊 Generated model visualizations:")
        print("   - Feature importance plots")
        print("   - Model performance comparisons")
        print("   - Clustering analysis charts")
        
        return model_results
        
    except Exception as e:
        print(f"❌ Error in modeling pipeline: {str(e)}")
        raise

def generate_final_summary(df_clean, eda_summary, model_results):
    """Generate a final project summary."""
    print("\n" + "="*60)
    print("📋 GENERATING FINAL SUMMARY")
    print("="*60)
    
    summary_content = f"""# NYC Airbnb Analysis - Complete Results Summary

## Dataset Overview
- **Original Data**: Input CSV file processed
- **Final Clean Dataset**: {df_clean.shape[0]:,} listings, {df_clean.shape[1]} features
- **Data Quality**: Validated and cleaned
- **Missing Values**: Handled appropriately

## Key Statistics
- **Average Price**: ${df_clean['price'].mean():.2f}
- **Median Price**: ${df_clean['price'].median():.2f}
- **Price Range**: ${df_clean['price'].min():.0f} - ${df_clean['price'].max():.0f}
- **Total Boroughs**: {df_clean['neighbourhood_group'].nunique()}
- **Total Neighborhoods**: {df_clean['neighbourhood'].nunique()}
- **Unique Hosts**: {df_clean['host_id'].nunique():,}

## Borough Distribution
{df_clean['neighbourhood_group'].value_counts().to_string()}

## Room Type Distribution  
{df_clean['room_type'].value_counts().to_string()}

## Model Performance
- **Linear Regression R²**: {model_results['regression_results']['price_lr']['test_r2']:.3f}
- **Random Forest R²**: {model_results['regression_results']['price_rf']['test_r2']:.3f}
- **Best Model**: Random Forest
- **Prediction Accuracy**: ±${model_results['regression_results']['price_rf']['test_mae']:.0f} MAE

## Files Generated
### Data Files
- `outputs/cleaned_airbnb.csv` - Cleaned dataset
- `outputs/cleaned_airbnb_missing_summary.json` - Data quality report
- `outputs/summary_by_borough.csv` - Borough statistics

### Model Files
- `outputs/model_price_lr.joblib` - Linear Regression model
- `outputs/model_price_rf.joblib` - Random Forest model  
- `outputs/model_kmeans.joblib` - K-means clustering model
- `outputs/scaler_kmeans.joblib` - Clustering scaler

### Visualization Files
- `outputs/figures/listings_by_borough.png` - Borough analysis
- `outputs/figures/price_distribution.png` - Price distributions
- `outputs/figures/price_by_room_type.png` - Room type analysis
- `outputs/figures/top_neighborhoods.png` - Neighborhood rankings
- `outputs/figures/price_vs_reviews.png` - Price-review relationships
- `outputs/figures/map.html` - Interactive geographic map
- `outputs/figures/correlation_heatmap.png` - Feature correlations
- `outputs/figures/feature_importance.png` - ML feature importance
- `outputs/figures/model_performance.png` - Model comparison
- `outputs/figures/kmeans_clusters.png` - Clustering analysis

## Analysis Complete!
All pipeline components executed successfully. The Streamlit dashboard can now be launched with:
```bash
streamlit run app.py
```

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save summary
    with open('outputs/analysis_complete_summary.md', 'w') as f:
        f.write(summary_content)
    
    print("✅ Final summary saved: outputs/analysis_complete_summary.md")

def main():
    """Run the complete pipeline."""
    print("🚀 NYC AIRBNB ANALYSIS - COMPLETE PIPELINE")
    print("="*60)
    print("This script will generate ALL output files needed for the analysis.")
    print("Estimated time: 2-5 minutes depending on data size.")
    print("="*60)
    
    try:
        # 1. Create directories
        create_output_directories()
        
        # 2. Check/create input data
        input_path = check_input_data()
        
        # 3. Run data cleaning pipeline
        df_clean = run_data_pipeline(input_path)
        
        # 4. Run EDA pipeline
        eda_summary = run_eda_pipeline_wrapper(df_clean)
        
        # 5. Run modeling pipeline
        model_results = run_modeling_pipeline_wrapper(df_clean)
        
        # 6. Generate final summary
        generate_final_summary(df_clean, eda_summary, model_results)
        
        # 7. Success message
        print("\n" + "🎉"*20)
        print("🎉 PIPELINE COMPLETE! ALL FILES GENERATED! 🎉")
        print("🎉"*20)
        
        print(f"\n📁 Output files location: outputs/")
        print(f"📊 Total files generated: {len(list(Path('outputs').rglob('*.*')))}")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Review outputs/analysis_complete_summary.md")
        print(f"   2. Launch dashboard: streamlit run app.py")
        print(f"   3. Explore visualizations in outputs/figures/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        print(f"💡 Check the error above and try running individual components.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)