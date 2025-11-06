"""
Machine Learning models for NYC Airbnb analysis.

This module contains functions for building predictive models including
price regression and clustering analysis to identify patterns and
make predictions on Airbnb listings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def prepare_features_for_modeling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features for machine learning models.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Features (X) and target (y) dataframes
    """
    # Create a copy for modeling
    df_model = df.copy()
    
    print(f"Input dataset shape: {df_model.shape}")
    print(f"Input columns: {list(df_model.columns)}")
    
    # Select features for modeling
    numeric_features = [
        'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365'
    ]
    
    # Add engineered features if they exist
    engineered_features = ['active_days', 'reviews_per_year', 'host_productivity']
    for feature in engineered_features:
        if feature in df_model.columns:
            numeric_features.append(feature)
    
    categorical_features = ['neighbourhood_group', 'room_type']
    
    # Prepare feature matrix
    X = pd.DataFrame()
    
    # Add numeric features that exist
    available_numeric = []
    for feature in numeric_features:
        if feature in df_model.columns:
            # Handle missing values
            if df_model[feature].isnull().sum() > 0:
                print(f"Filling {df_model[feature].isnull().sum()} missing values in {feature}")
                df_model[feature] = df_model[feature].fillna(df_model[feature].median())
            
            X[feature] = df_model[feature]
            available_numeric.append(feature)
        else:
            print(f"Warning: Feature {feature} not found in dataset")
    
    # One-hot encode categorical features
    for feature in categorical_features:
        if feature in df_model.columns:
            dummies = pd.get_dummies(df_model[feature], prefix=feature, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
            print(f"One-hot encoded {feature}: {len(dummies.columns)} new columns")
        else:
            print(f"Warning: Categorical feature {feature} not found in dataset")
    
    # Target variable
    y = df_model['price']
    
    # Validate data
    print(f"\nData validation:")
    print(f"Features prepared: {X.shape[1]} features, {len(X)} samples")
    print(f"Target variable range: ${y.min():.2f} - ${y.max():.2f}")
    print(f"Target variable mean: ${y.mean():.2f}")
    print(f"Missing values in X: {X.isnull().sum().sum()}")
    print(f"Missing values in y: {y.isnull().sum()}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Remove any remaining missing values
    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
        print("Removing rows with missing values...")
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        print(f"Final shape after removing missing: X={X.shape}, y={y.shape}")
    
    return X, y


def train_price_regression_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train and evaluate price regression models.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (price)
        
    Returns:
        Dict[str, Any]: Dictionary containing trained models and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    models = {}
    results = {}
    
    # 1. Linear Regression
    print("\n=== TRAINING LINEAR REGRESSION ===")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Predictions
    lr_train_pred = lr_model.predict(X_train)
    lr_test_pred = lr_model.predict(X_test)
    
    # Metrics
    lr_metrics = {
        'train_mae': mean_absolute_error(y_train, lr_train_pred),
        'test_mae': mean_absolute_error(y_test, lr_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, lr_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, lr_test_pred)),
        'train_r2': r2_score(y_train, lr_train_pred),
        'test_r2': r2_score(y_test, lr_test_pred)
    }
    
    models['price_lr'] = lr_model
    results['price_lr'] = lr_metrics
    
    print(f"Linear Regression Results:")
    print(f"  Train R²: {lr_metrics['train_r2']:.3f}")
    print(f"  Test R²: {lr_metrics['test_r2']:.3f}")
    print(f"  Test MAE: ${lr_metrics['test_mae']:.2f}")
    print(f"  Test RMSE: ${lr_metrics['test_rmse']:.2f}")
    
    # Debug negative R²
    if lr_metrics['test_r2'] < 0:
        print(f"  ⚠️  Negative R² detected! Model is worse than predicting the mean.")
        print(f"  Target mean: ${y_test.mean():.2f}")
        print(f"  Prediction mean: ${lr_test_pred.mean():.2f}")
        print(f"  Target std: ${y_test.std():.2f}")
        print(f"  Prediction std: ${lr_test_pred.std():.2f}")
    
    # 2. Random Forest Regressor
    print("\n=== TRAINING RANDOM FOREST ===")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Predictions
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    # Metrics
    rf_metrics = {
        'train_mae': mean_absolute_error(y_train, rf_train_pred),
        'test_mae': mean_absolute_error(y_test, rf_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, rf_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, rf_test_pred)),
        'train_r2': r2_score(y_train, rf_train_pred),
        'test_r2': r2_score(y_test, rf_test_pred)
    }
    
    models['price_rf'] = rf_model
    results['price_rf'] = rf_metrics
    
    print(f"Random Forest Results:")
    print(f"  Train R²: {rf_metrics['train_r2']:.3f}")
    print(f"  Test R²: {rf_metrics['test_r2']:.3f}")
    print(f"  Test MAE: ${rf_metrics['test_mae']:.2f}")
    print(f"  Test RMSE: ${rf_metrics['test_rmse']:.2f}")
    
    # Debug negative R²
    if rf_metrics['test_r2'] < 0:
        print(f"  ⚠️  Negative R² detected! Model is worse than predicting the mean.")
        print(f"  Target mean: ${y_test.mean():.2f}")
        print(f"  Prediction mean: ${rf_test_pred.mean():.2f}")
        print(f"  Target std: ${y_test.std():.2f}")
        print(f"  Prediction std: ${rf_test_pred.std():.2f}")
    
    # Cross-validation for Random Forest
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    print(f"  CV R² Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Store test data for plotting
    results['test_data'] = {
        'X_test': X_test,
        'y_test': y_test,
        'lr_pred': lr_test_pred,
        'rf_pred': rf_test_pred
    }
    
    return models, results


def plot_feature_importance(model, feature_names: list, 
                          save_path: str = 'outputs/figures/feature_importance.png') -> None:
    """
    Plot feature importance for Random Forest model.
    
    Args:
        model: Trained Random Forest model
        feature_names (list): List of feature names
        save_path (str): Path to save the plot
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Select top 15 features for readability
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Create horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = plt.barh(range(top_n), top_importances, color=colors)
    
    plt.yticks(range(top_n), top_features)
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


def plot_model_performance(results: Dict, save_path: str = 'outputs/figures/model_performance.png') -> None:
    """
    Plot model performance comparison and prediction accuracy.
    
    Args:
        results (Dict): Results dictionary from model training
        save_path (str): Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Model comparison bar chart
    models = ['Linear Regression', 'Random Forest']
    test_r2 = [results['price_lr']['test_r2'], results['price_rf']['test_r2']]
    test_mae = [results['price_lr']['test_mae'], results['price_rf']['test_mae']]
    
    x_pos = np.arange(len(models))
    bars1 = ax1.bar(x_pos, test_r2, color=['skyblue', 'lightcoral'], alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance Comparison (R²)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, max(test_r2) * 1.1)
    
    # Add value labels
    for bar, score in zip(bars1, test_r2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. MAE comparison
    bars2 = ax2.bar(x_pos, test_mae, color=['skyblue', 'lightcoral'], alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Mean Absolute Error ($)')
    ax2.set_title('Model Performance Comparison (MAE)', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models)
    
    # Add value labels
    for bar, mae in zip(bars2, test_mae):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'${mae:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Actual vs Predicted (Linear Regression)
    test_data = results['test_data']
    ax3.scatter(test_data['y_test'], test_data['lr_pred'], alpha=0.5, s=20, color='skyblue')
    ax3.plot([test_data['y_test'].min(), test_data['y_test'].max()], 
             [test_data['y_test'].min(), test_data['y_test'].max()], 'r--', lw=2)
    ax3.set_xlabel('Actual Price ($)')
    ax3.set_ylabel('Predicted Price ($)')
    ax3.set_title('Linear Regression: Actual vs Predicted', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add R² score
    lr_r2 = results['price_lr']['test_r2']
    ax3.text(0.05, 0.95, f'R² = {lr_r2:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Actual vs Predicted (Random Forest)
    ax4.scatter(test_data['y_test'], test_data['rf_pred'], alpha=0.5, s=20, color='lightcoral')
    ax4.plot([test_data['y_test'].min(), test_data['y_test'].max()], 
             [test_data['y_test'].min(), test_data['y_test'].max()], 'r--', lw=2)
    ax4.set_xlabel('Actual Price ($)')
    ax4.set_ylabel('Predicted Price ($)')
    ax4.set_title('Random Forest: Actual vs Predicted', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add R² score
    rf_r2 = results['price_rf']['test_r2']
    ax4.text(0.05, 0.95, f'R² = {rf_r2:.3f}', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model performance plot saved to {save_path}")


def perform_clustering_analysis(df: pd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
    """
    Perform K-means clustering analysis on selected features.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        n_clusters (int): Number of clusters
        
    Returns:
        Dict[str, Any]: Clustering results and model
    """
    print(f"\n=== PERFORMING K-MEANS CLUSTERING (k={n_clusters}) ===")
    
    # Select features for clustering
    clustering_features = ['price', 'availability_365', 'number_of_reviews', 
                          'minimum_nights', 'calculated_host_listings_count']
    
    # Prepare data
    X_cluster = df[clustering_features].copy()
    
    # Handle any remaining missing values
    X_cluster = X_cluster.fillna(X_cluster.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Calculate cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=clustering_features)
    centers_df.index = [f'Cluster_{i}' for i in range(n_clusters)]
    
    # Calculate cluster statistics
    cluster_stats = df_clustered.groupby('cluster').agg({
        'price': ['count', 'mean', 'median'],
        'availability_365': 'mean',
        'number_of_reviews': 'mean',
        'minimum_nights': 'mean',
        'calculated_host_listings_count': 'mean',
        'neighbourhood_group': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed',
        'room_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
    }).round(2)
    
    print("Cluster Analysis Complete!")
    print(f"Cluster sizes: {pd.Series(cluster_labels).value_counts().sort_index().values}")
    
    results = {
        'model': kmeans,
        'scaler': scaler,
        'labels': cluster_labels,
        'centers': centers_df,
        'statistics': cluster_stats,
        'features': clustering_features,
        'data': df_clustered
    }
    
    return results


def plot_clustering_results(clustering_results: Dict, 
                          save_path: str = 'outputs/figures/kmeans_clusters.png') -> None:
    """
    Plot clustering results with multiple visualizations.
    
    Args:
        clustering_results (Dict): Results from clustering analysis
        save_path (str): Path to save the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    df_clustered = clustering_results['data']
    n_clusters = len(clustering_results['centers'])
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    # 1. Price vs Availability colored by cluster
    for i in range(n_clusters):
        cluster_data = df_clustered[df_clustered['cluster'] == i]
        ax1.scatter(cluster_data['availability_365'], cluster_data['price'], 
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=20)
    
    ax1.set_xlabel('Availability (days/year)')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Clusters: Price vs Availability', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price vs Number of Reviews colored by cluster
    for i in range(n_clusters):
        cluster_data = df_clustered[df_clustered['cluster'] == i]
        ax2.scatter(cluster_data['number_of_reviews'], cluster_data['price'], 
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=20)
    
    ax2.set_xlabel('Number of Reviews')
    ax2.set_ylabel('Price ($)')
    ax2.set_title('Clusters: Price vs Reviews', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cluster sizes
    cluster_sizes = df_clustered['cluster'].value_counts().sort_index()
    bars = ax3.bar(range(n_clusters), cluster_sizes.values, color=colors, alpha=0.8)
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Number of Listings')
    ax3.set_title('Cluster Sizes', fontweight='bold')
    ax3.set_xticks(range(n_clusters))
    ax3.set_xticklabels([f'Cluster {i}' for i in range(n_clusters)])
    
    # Add value labels
    for bar, size in zip(bars, cluster_sizes.values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Cluster centers heatmap
    centers_df = clustering_results['centers']
    # Normalize for better visualization
    centers_normalized = (centers_df - centers_df.min()) / (centers_df.max() - centers_df.min())
    
    im = ax4.imshow(centers_normalized.T, cmap='RdYlBu_r', aspect='auto')
    ax4.set_xticks(range(n_clusters))
    ax4.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    ax4.set_yticks(range(len(centers_df.columns)))
    ax4.set_yticklabels(centers_df.columns, rotation=45, ha='right')
    ax4.set_title('Cluster Centers (Normalized)', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Normalized Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Clustering results plot saved to {save_path}")


def save_models(models: Dict, save_dir: str = 'outputs/') -> None:
    """
    Save trained models to disk.
    
    Args:
        models (Dict): Dictionary of trained models
        save_dir (str): Directory to save models
    """
    for model_name, model in models.items():
        model_path = f"{save_dir}model_{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")


def save_all_models(regression_models: Dict, clustering_results: Dict, feature_names: list, save_dir: str = 'outputs/') -> None:
    """
    Save all trained models and preprocessing objects to disk as .joblib files.
    
    Args:
        regression_models (Dict): Dictionary of trained regression models
        clustering_results (Dict): Dictionary containing clustering model and scaler
        feature_names (list): List of feature names used in regression models
        save_dir (str): Directory to save models
    """
    print("\n=== SAVING ALL MODELS AS .JOBLIB FILES ===")
    
    # Save regression models
    for model_name, model in regression_models.items():
        model_path = f"{save_dir}model_{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"✓ Regression model saved: {model_path}")
    
    # Save regression feature names
    regression_features_path = f"{save_dir}regression_features.joblib"
    joblib.dump(feature_names, regression_features_path)
    print(f"✓ Regression features saved: {regression_features_path}")
    
    # Save clustering model
    if 'model' in clustering_results:
        kmeans_path = f"{save_dir}model_kmeans.joblib"
        joblib.dump(clustering_results['model'], kmeans_path)
        print(f"✓ K-means model saved: {kmeans_path}")
    
    # Save clustering scaler
    if 'scaler' in clustering_results:
        scaler_path = f"{save_dir}scaler_kmeans.joblib"
        joblib.dump(clustering_results['scaler'], scaler_path)
        print(f"✓ K-means scaler saved: {scaler_path}")
    
    # Save cluster centers as joblib for consistency
    if 'centers' in clustering_results:
        centers_path = f"{save_dir}cluster_centers.joblib"
        joblib.dump(clustering_results['centers'], centers_path)
        print(f"✓ Cluster centers saved: {centers_path}")
    
    # Save clustering feature names for model inference
    if 'features' in clustering_results:
        clustering_features_path = f"{save_dir}clustering_features.joblib"
        joblib.dump(clustering_results['features'], clustering_features_path)
        print(f"✓ Clustering features saved: {clustering_features_path}")
    
    # Create and save model inventory
    model_inventory = {
        'regression_models': list(regression_models.keys()),
        'clustering_model': 'kmeans' if 'model' in clustering_results else None,
        'regression_features': feature_names,
        'clustering_features': clustering_results.get('features', []),
        'total_models': len(regression_models) + (1 if 'model' in clustering_results else 0),
        'files_created': [
            f"model_{name}.joblib" for name in regression_models.keys()
        ] + [
            'model_kmeans.joblib',
            'scaler_kmeans.joblib', 
            'cluster_centers.joblib',
            'regression_features.joblib',
            'clustering_features.joblib'
        ]
    }
    
    inventory_path = f"{save_dir}model_inventory.joblib"
    joblib.dump(model_inventory, inventory_path)
    print(f"✓ Model inventory saved: {inventory_path}")
    
    print(f"\n📊 MODEL SUMMARY:")
    print(f"   • Regression models: {len(regression_models)}")
    print(f"   • Clustering models: {1 if 'model' in clustering_results else 0}")
    print(f"   • Total .joblib files: {len(model_inventory['files_created'])}")
    print("All models successfully saved as .joblib files!")


def run_modeling_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the complete modeling pipeline.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        Dict[str, Any]: Complete modeling results
    """
    print("=== STARTING MODELING PIPELINE ===")
    
    # 1. Prepare features
    X, y = prepare_features_for_modeling(df)
    
    # 2. Train regression models
    models, regression_results = train_price_regression_models(X, y)
    
    # 3. Plot feature importance (Random Forest)
    plot_feature_importance(models['price_rf'], list(X.columns))
    
    # 4. Plot model performance
    plot_model_performance(regression_results)
    
    # 5. Perform clustering
    clustering_results = perform_clustering_analysis(df, n_clusters=5)
    
    # 6. Plot clustering results
    plot_clustering_results(clustering_results)
    
    # 7. Save all models as .joblib files
    save_all_models(models, clustering_results, list(X.columns))
    
    print("\n=== MODELING PIPELINE COMPLETE ===")
    
    # Compile results
    results = {
        'regression_models': models,
        'regression_results': regression_results,
        'clustering_results': clustering_results,
        'features': list(X.columns)
    }
    
    return results