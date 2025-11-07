# Technical Architecture Guide
## Model Creation, Storage, and Integration

---

## Table of Contents
1. [Model Creation in model.py](#model-creation)
2. [Joblib File Format](#joblib-format)
3. [Model Integration in app.py](#model-integration)
4. [Map Generation Process](#map-generation)
5. [Complete Data Flow](#data-flow)

---

## 1. Model Creation in model.py {#model-creation}

### Step-by-Step Process

#### Step 1: Import Required Libraries
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
```

**What these do:**
- `RandomForestRegressor`: The Random Forest algorithm
- `LinearRegression`: The Linear Regression algorithm
- `joblib`: Library to save/load Python objects efficiently

#### Step 2: Prepare Features (X) and Target (y)

**Location in code:** `src/model.py` → `prepare_features_for_modeling()`

```python
def prepare_features_for_modeling(df):
    # Select numeric features
    numeric_features = [
        'latitude', 'longitude', 'minimum_nights', 
        'number_of_reviews', 'reviews_per_month',
        'calculated_host_listings_count', 'availability_365',
        'active_days', 'reviews_per_year', 'host_productivity'
    ]
    
    # Select categorical features
    categorical_features = ['neighbourhood_group', 'room_type']
    
    # Create feature matrix X
    X = pd.DataFrame()
    
    # Add numeric features
    for feature in numeric_features:
        X[feature] = df[feature]
    
    # One-hot encode categorical features
    for feature in categorical_features:
        dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
        X = pd.concat([X, dummies], axis=1)
    
    # Target variable
    y = df['price']
    
    return X, y
```

**Result:**
- X: 48,884 rows × 16 columns (features)
- y: 48,884 values (prices)

#### Step 3: Split Data into Train and Test Sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing
    random_state=42  # For reproducibility
)
```

**Result:**
- Training: 39,107 listings (80%)
- Testing: 9,777 listings (20%)

#### Step 4: Train Linear Regression Model

```python
# Create model instance
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)
```

**What happens internally:**

1. Calculates optimal weights for each feature
2. Finds the best-fit line: `price = w₁×feature₁ + w₂×feature₂ + ... + b`
3. Stores these weights inside the `lr_model` object

**Model object now contains:**
- Learned weights (coefficients) for each feature
- Intercept (bias term)
- Methods to make predictions

#### Step 5: Train Random Forest Model

```python
# Create model instance with parameters
rf_model = RandomForestRegressor(
    n_estimators=100,      # Create 100 decision trees
    max_depth=15,          # Each tree max 15 levels deep
    min_samples_split=5,   # Need 5+ samples to split node
    min_samples_leaf=2,    # Each leaf needs 2+ samples
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores
)

# Train the model
rf_model.fit(X_train, y_train)
```

**What happens internally:**
1. Creates 100 empty decision trees
2. For each tree:
   - Randomly samples data (bootstrap sampling)
   - Randomly selects features
   - Builds a decision tree
   - Stores the tree structure
3. All 100 trees are stored in `rf_model` object

**Model object now contains:**
- 100 complete decision tree structures
- Feature importance scores
- Methods to make predictions (averages all trees)

#### Step 6: Save Models as .joblib Files

**Location in code:** `src/model.py` → `save_all_models()`

```python
import joblib

def save_all_models(regression_models, clustering_results, feature_names, save_dir='outputs/'):
    # Save Linear Regression model
    joblib.dump(regression_models['price_lr'], 'outputs/model_price_lr.joblib')
    
    # Save Random Forest model
    joblib.dump(regression_models['price_rf'], 'outputs/model_price_rf.joblib')
    
    # Save feature names (important for predictions!)
    joblib.dump(feature_names, 'outputs/regression_features.joblib')
    
    # Save clustering model
    joblib.dump(clustering_results['model'], 'outputs/model_kmeans.joblib')
    
    # Save clustering scaler
    joblib.dump(clustering_results['scaler'], 'outputs/scaler_kmeans.joblib')
```

**What gets saved in each .joblib file:**

**model_price_lr.joblib:**
```
LinearRegression object containing:
├── coef_ (array of 16 weights, one per feature)
├── intercept_ (bias term)
├── n_features_in_ (16)
└── feature_names_in_ (list of feature names)
```

**model_price_rf.joblib:**
```
RandomForestRegressor object containing:
├── estimators_ (list of 100 DecisionTreeRegressor objects)
├── feature_importances_ (array showing importance of each feature)
├── n_estimators (100)
├── max_depth (15)
├── n_features_in_ (16)
└── All 100 decision tree structures with:
    ├── Tree nodes
    ├── Split conditions
    ├── Leaf values
    └── Feature indices used
```

**regression_features.joblib:**
```
List of feature names in exact order:
[
    'latitude',
    'longitude',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365',
    'active_days',
    'reviews_per_year',
    'host_productivity',
    'neighbourhood_group_Brooklyn',
    'neighbourhood_group_Manhattan',
    'neighbourhood_group_Queens',
    'neighbourhood_group_Staten Island',
    'room_type_Private room',
    'room_type_Shared room'
]
```

---

## 2. Joblib File Format {#joblib-format}

### What is Joblib?

**Joblib** is a Python library that efficiently saves and loads Python objects.

### Why Joblib instead of Pickle?

| Feature | Joblib | Pickle |
|---------|--------|--------|
| Speed for large arrays | ⚡ Fast | 🐌 Slow |
| Compression | ✅ Built-in | ❌ Manual |
| NumPy arrays | ✅ Optimized | ⚠️ Basic |
| File size | 📦 Smaller | 📦 Larger |
| Best for | ML models | General objects |

### How Joblib Works

```python
# SAVING
import joblib

# Save any Python object
joblib.dump(my_model, 'model.joblib')

# What happens:
# 1. Serializes the object (converts to bytes)
# 2. Compresses the data
# 3. Writes to disk

# LOADING
loaded_model = joblib.load('model.joblib')

# What happens:
# 1. Reads file from disk
# 2. Decompresses the data
# 3. Deserializes (reconstructs the object)
# 4. Returns the exact same object
```

### File Structure

**model_price_rf.joblib internal structure:**
```
Compressed binary file containing:
├── Metadata
│   ├── Python version
│   ├── Scikit-learn version
│   └── Object type (RandomForestRegressor)
├── Model parameters
│   ├── n_estimators: 100
│   ├── max_depth: 15
│   └── random_state: 42
├── Trained data
│   ├── 100 decision trees (serialized)
│   ├── Feature importances array
│   └── Training metadata
└── Methods (functions)
    ├── predict()
    ├── predict_proba()
    └── score()
```

**File sizes:**
- `model_price_lr.joblib`: ~5 KB (simple model)
- `model_price_rf.joblib`: ~50 MB (100 trees with full structure)
- `regression_features.joblib`: ~1 KB (just a list)

---

## 3. Model Integration in app.py {#model-integration}

### Step-by-Step Integration

#### Step 1: Load Models at App Startup

**Location:** `app.py` → `load_models()` function

```python
@st.cache_data
def load_models():
    """Load or create trained models."""
    try:
        # Load the Random Forest model
        rf_model = joblib.load('outputs/model_price_rf.joblib')
        
        # Return in a dictionary
        return {'random_forest': rf_model}
        
    except FileNotFoundError:
        # If models don't exist, try to train them
        if PIPELINE_AVAILABLE:
            st.info("Training models... This may take a moment.")
            df = pd.read_csv('outputs/cleaned_airbnb.csv')
            model_results = run_modeling_pipeline(df)
            
            # Load the newly trained model
            rf_model = joblib.load('outputs/model_price_rf.joblib')
            st.success("Models trained successfully!")
            return {'random_forest': rf_model}
        
        st.warning("Models not available.")
        return None
```

**What `@st.cache_data` does:**
- Loads the model only ONCE when app starts
- Caches it in memory
- Reuses the cached model for all users
- Doesn't reload on every page refresh

**Memory after loading:**
```
Streamlit App Memory:
├── Cached Data
│   ├── df (cleaned dataset) - ~50 MB
│   └── models
│       └── random_forest (RandomForestRegressor) - ~50 MB
└── Session State (per user)
    └── User-specific data
```

#### Step 2: Using Models for Predictions

**Location:** `app.py` → `prediction_tab()` function

```python
def prediction_tab(filtered_df, models):
    # User inputs
    pred_borough = st.selectbox("Borough", [...])
    pred_room_type = st.selectbox("Room Type", [...])
    pred_min_nights = st.number_input("Minimum Nights", ...)
    # ... more inputs
    
    if st.button("🎯 Predict Price"):
        # Step 1: Create feature vector in EXACT same order as training
        features = np.array([[
            pred_lat,                    # latitude
            pred_lon,                    # longitude
            pred_min_nights,             # minimum_nights
            pred_reviews,                # number_of_reviews
            0.0,                         # reviews_per_month (simplified)
            pred_host_listings,          # calculated_host_listings_count
            pred_availability,           # availability_365
            365 - pred_availability,     # active_days
            0.0,                         # reviews_per_year (simplified)
            pred_host_listings,          # host_productivity
            1 if pred_borough == 'Brooklyn' else 0,      # neighbourhood_group_Brooklyn
            1 if pred_borough == 'Manhattan' else 0,     # neighbourhood_group_Manhattan
            1 if pred_borough == 'Queens' else 0,        # neighbourhood_group_Queens
            1 if pred_borough == 'Staten Island' else 0, # neighbourhood_group_Staten Island
            1 if pred_room_type == 'Private room' else 0,  # room_type_Private room
            1 if pred_room_type == 'Shared room' else 0    # room_type_Shared room
        ]])
        
        # Step 2: Make prediction
        predicted_price = models['random_forest'].predict(features)[0]
        
        # Step 3: Display result
        st.success(f"💰 **Predicted Price: ${predicted_price:.0f}/night**")
```

**What happens during prediction:**

1. **Feature Vector Creation:**
```
Input: Borough=Manhattan, Room=Entire home, MinNights=2, ...

Feature Vector (1 row × 16 columns):
[40.7589, -73.9851, 2, 10, 0.0, 1, 180, 185, 0.0, 1, 0, 1, 0, 0, 0, 0]
 ↑        ↑        ↑  ↑   ↑    ↑  ↑    ↑    ↑    ↑  ↑  ↑  ↑  ↑  ↑  ↑
 lat      lon      mn rev rpm  hl avl  act  rpy  hp br ma qu si pr sr
```

2. **Random Forest Prediction Process:**
```python
# Internally, the model does:
predictions = []
for tree in rf_model.estimators_:  # Loop through 100 trees
    tree_prediction = tree.predict(features)
    predictions.append(tree_prediction)

# Average all predictions
final_prediction = np.mean(predictions)
```

3. **Result:**
```
Tree 1 predicts: $145
Tree 2 predicts: $152
Tree 3 predicts: $148
...
Tree 100 predicts: $150

Average: $148.50
```

---

## 4. Map Generation Process {#map-generation}

### Complete Map Generation Workflow

**Location:** `app.py` → `geographic_tab()` function

#### Step 1: Prepare Map Data

```python
def geographic_tab(filtered_df):
    st.subheader("🗺️ Geographic Distribution")
    
    if len(filtered_df) > 0:
        # User controls how many listings to show
        max_listings_on_map = st.slider(
            "Number of listings to show on map",
            min_value=1000,
            max_value=min(50000, len(filtered_df)),
            value=min(10000, len(filtered_df)),
            step=1000
        )
        
        # Sample data if needed (for performance)
        if len(filtered_df) > max_listings_on_map:
            map_df = filtered_df.sample(n=max_listings_on_map, random_state=42)
        else:
            map_df = filtered_df
```

**Why sampling?**
- Rendering 48,000+ points on a map is slow
- Browser can freeze with too many markers
- 10,000 points is a good balance

#### Step 2: Create Interactive Map with Plotly

```python
import plotly.express as px

fig_map = px.scatter_mapbox(
    map_df,                              # Data to plot
    lat="latitude",                      # Column for latitude
    lon="longitude",                     # Column for longitude
    color="price",                       # Color points by price
    size="number_of_reviews",            # Size points by reviews
    hover_data=["neighbourhood_group", "room_type", "price"],  # Tooltip info
    color_continuous_scale="Viridis",    # Color scheme
    size_max=15,                         # Maximum marker size
    zoom=10,                             # Initial zoom level
    title=f"Airbnb Listings Map (showing {len(map_df):,} of {len(filtered_df):,} listings)"
)
```

**What each parameter does:**

**lat & lon:**
```python
# Each listing has coordinates
Listing 1: lat=40.7589, lon=-73.9851 (Manhattan)
Listing 2: lat=40.6782, lon=-73.9442 (Brooklyn)
...

# Plotly plots each as a point on the map
```

**color="price":**
```python
# Creates a color gradient based on price
Low price ($50)  → Dark purple
Medium ($150)    → Green/Yellow
High price ($300)→ Bright yellow

# Automatically creates a color scale legend
```

**size="number_of_reviews":**
```python
# Marker size proportional to reviews
0 reviews   → Tiny dot (size 1)
50 reviews  → Medium dot (size 8)
200 reviews → Large dot (size 15)

# More popular listings = Bigger markers
```

**hover_data:**
```python
# When user hovers over a point, shows:
"""
Neighbourhood Group: Manhattan
Room Type: Entire home/apt
Price: $150
"""
```

#### Step 3: Configure Map Style

```python
fig_map.update_layout(
    mapbox_style="open-street-map",  # Use OpenStreetMap tiles
    height=600,                       # Map height in pixels
    margin={"r":0, "t":50, "l":0, "b":0}  # Remove margins
)
```

**Available map styles:**
- `"open-street-map"`: Free, detailed street map
- `"carto-positron"`: Light, minimal style
- `"carto-darkmatter"`: Dark theme
- `"stamen-terrain"`: Topographic
- `"stamen-toner"`: Black and white

#### Step 4: Render Map in Streamlit

```python
st.plotly_chart(fig_map, use_container_width=True)
```

**What happens:**
1. Plotly generates HTML/JavaScript code
2. Streamlit embeds it in the web page
3. User's browser renders the interactive map
4. User can:
   - Zoom in/out
   - Pan around
   - Hover for details
   - Click markers

### Map Rendering Architecture

```
┌─────────────────────────────────────────┐
│         Streamlit Server (Python)        │
│                                          │
│  1. Load data from cleaned_airbnb.csv   │
│  2. Filter based on user selections      │
│  3. Sample 10,000 listings               │
│  4. Create Plotly figure                 │
│  5. Convert to HTML/JavaScript           │
│                                          │
│         ↓ Send HTML to browser           │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         User's Web Browser               │
│                                          │
│  1. Receive HTML/JavaScript              │
│  2. Load OpenStreetMap tiles             │
│  3. Render 10,000 markers                │
│  4. Enable interactivity                 │
│     - Zoom/Pan                           │
│     - Hover tooltips                     │
│     - Click events                       │
│                                          │
└─────────────────────────────────────────┘
```

### Map Data Structure

**What gets sent to the browser:**

```javascript
{
  data: [
    {
      type: 'scattermapbox',
      lat: [40.7589, 40.6782, ...],  // 10,000 latitudes
      lon: [-73.9851, -73.9442, ...], // 10,000 longitudes
      marker: {
        color: [150, 100, 200, ...],  // Prices for coloring
        size: [10, 5, 15, ...],       // Review counts for sizing
        colorscale: 'Viridis'
      },
      hovertemplate: '<b>%{customdata[0]}</b><br>...'
    }
  ],
  layout: {
    mapbox: {
      style: 'open-street-map',
      center: {lat: 40.7128, lon: -74.0060},
      zoom: 10
    },
    height: 600
  }
}
```

---

## 5. Complete Data Flow {#data-flow}

### End-to-End Process

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Data Collection                                      │
│ File: notebooks/AB_NYC_2019.csv                              │
│ Size: 48,895 listings × 16 features                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Data Cleaning (src/data_prep.py)                    │
│ - Handle missing values                                      │
│ - Remove duplicates                                          │
│ - Filter outliers                                            │
│ - Feature engineering                                        │
│ Output: outputs/cleaned_airbnb.csv (48,884 listings)         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: Model Training (src/model.py)                       │
│ - Prepare features (X) and target (y)                        │
│ - Split train/test (80/20)                                   │
│ - Train Linear Regression                                    │
│ - Train Random Forest (100 trees)                            │
│ - Evaluate performance                                       │
│ - Save models as .joblib files                               │
│                                                              │
│ Outputs:                                                     │
│ ├── model_price_lr.joblib (~5 KB)                           │
│ ├── model_price_rf.joblib (~50 MB)                          │
│ ├── regression_features.joblib (~1 KB)                      │
│ └── model_kmeans.joblib (~1 MB)                             │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Streamlit App Startup (app.py)                      │
│ - Load cleaned data (cached)                                 │
│ - Load trained models (cached)                               │
│ - Initialize UI components                                   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: User Interaction                                     │
│                                                              │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ User Action: Apply Filters                             │  │
│ │ - Select borough: Manhattan                            │  │
│ │ - Select room type: Entire home/apt                    │  │
│ │ - Price range: $100-$300                               │  │
│ └────────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ App Response: Filter Data                              │  │
│ │ filtered_df = df[                                      │  │
│ │     (df['neighbourhood_group'] == 'Manhattan') &       │  │
│ │     (df['room_type'] == 'Entire home/apt') &           │  │
│ │     (df['price'].between(100, 300))                    │  │
│ │ ]                                                      │  │
│ │ Result: 5,234 listings                                 │  │
│ └────────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ Generate Map                                           │  │
│ │ 1. Sample 5,234 listings (all fit in limit)           │  │
│ │ 2. Create Plotly scatter_mapbox                        │  │
│ │ 3. Render interactive map                              │  │
│ └────────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ User Action: Request Price Prediction                  │  │
│ │ Inputs:                                                │  │
│ │ - Borough: Manhattan                                   │  │
│ │ - Room: Entire home/apt                                │  │
│ │ - Min nights: 2                                        │  │
│ │ - Reviews: 50                                          │  │
│ │ - Availability: 200 days                               │  │
│ └────────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ App Response: Make Prediction                          │  │
│ │ 1. Create feature vector (16 values)                   │  │
│ │ 2. Call model.predict(features)                        │  │
│ │ 3. Random Forest averages 100 tree predictions         │  │
│ │ 4. Return: $185/night                                  │  │
│ │ 5. Display with confidence interval: $136-$234         │  │
│ └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### File Dependencies

```
Project Structure:
├── notebooks/
│   └── AB_NYC_2019.csv ──────────┐
│                                  │
├── src/                           │
│   ├── data_prep.py ──────────────┤
│   │   (reads CSV)                │
│   │   (outputs cleaned CSV)      │
│   │                              ↓
│   └── model.py ──────────────────┤
│       (reads cleaned CSV)        │
│       (outputs .joblib files)    │
│                                  │
├── outputs/                       │
│   ├── cleaned_airbnb.csv ←───────┤
│   ├── model_price_lr.joblib      │
│   ├── model_price_rf.joblib      │
│   └── regression_features.joblib │
│                                  │
├── app.py ───────────────────────┘
│   (reads cleaned CSV)
│   (loads .joblib models)
│   (generates interactive UI)
│
└── run_complete_pipeline.py
    (orchestrates everything)
```

### Memory Usage During Runtime

```
Streamlit App Memory Footprint:

┌─────────────────────────────────────┐
│ Cached Data (Shared across users)  │
├─────────────────────────────────────┤
│ cleaned_airbnb.csv    ~50 MB       │
│ model_price_rf        ~50 MB       │
│ model_price_lr        ~0.5 MB      │
│ Plotly map data       ~10 MB       │
├─────────────────────────────────────┤
│ Total Cached:         ~110 MB      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Per-User Session (Separate)        │
├─────────────────────────────────────┤
│ Filtered dataframe    ~5 MB        │
│ UI state              ~1 MB        │
│ Temporary variables   ~2 MB        │
├─────────────────────────────────────┤
│ Per User:             ~8 MB        │
└─────────────────────────────────────┘

Total for 10 concurrent users:
110 MB (cached) + 10 × 8 MB (sessions) = 190 MB
```

---

## Key Takeaways

### Model Creation (.joblib files)
1. **Training:** Models learn patterns from 39,107 listings
2. **Serialization:** Trained models converted to binary format
3. **Compression:** Joblib compresses for efficient storage
4. **Storage:** Saved as .joblib files (~50 MB for Random Forest)

### Model Integration (app.py)
1. **Loading:** Joblib deserializes .joblib files back to Python objects
2. **Caching:** Streamlit caches models in memory (load once, use many times)
3. **Prediction:** User inputs → Feature vector → Model.predict() → Result
4. **Display:** Results shown in Streamlit UI

### Map Generation
1. **Data Prep:** Filter and sample listings for performance
2. **Plotly:** Create interactive scatter_mapbox figure
3. **Rendering:** Convert to HTML/JavaScript
4. **Browser:** User's browser renders the interactive map
5. **Interaction:** Zoom, pan, hover all handled client-side

---

**This architecture enables:**
- ✅ Fast predictions (models loaded once, cached)
- ✅ Interactive maps (Plotly + OpenStreetMap)
- ✅ Scalable (handles 48,000+ listings)
- ✅ User-friendly (Streamlit UI)
- ✅ Maintainable (modular code structure)
