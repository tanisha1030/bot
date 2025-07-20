import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
import os

# Configure the page
st.set_page_config(page_title="Botnet Detection", layout="wide")

# Title
st.title("🤖 Botnet Detection in Robotic Network Logs")

# Load data function - Updated to use the actual data path from main.py
@st.cache_data
def load_data():
    """Load the dataset from the same path used in main.py"""
    try:
        # Primary path from main.py
        primary_path = "/content/synthetic_robot_logs.csv"
        
        # Alternative paths to try
        possible_paths = [
            primary_path,
            "synthetic_robot_logs.csv",
            "data/synthetic_robot_logs.csv",
            "./synthetic_robot_logs.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"✅ Dataset loaded from {path}")
                
                # Validate the data structure matches main.py expectations
                expected_columns = ['packet_size', 'interval', 'is_botnet']
                if all(col in df.columns for col in expected_columns):
                    # Display basic info like in main.py
                    st.info(f"""
                    📊 **Dataset Information (from main.py path):**
                    - Shape: {df.shape}
                    - Class distribution: {df['is_botnet'].value_counts().to_dict()}
                    - Missing values: {df.isnull().sum().sum()}
                    """)
                    return df
                else:
                    st.warning(f"⚠️ Dataset found but missing required columns. Expected: {expected_columns}")
        
        # If no valid file found, inform user
        st.error(f"""
        ❌ **Dataset not found at expected paths:**
        - Primary: {primary_path}
        - Alternatives: {possible_paths[1:]}
        
        **To use this app with your actual data:**
        1. Ensure 'synthetic_robot_logs.csv' exists in one of the above paths
        2. Make sure it has columns: packet_size, interval, is_botnet
        3. Or upload the file to the same location as main.py
        """)
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# Load pre-trained model from main.py if available
@st.cache_resource
def load_pretrained_model():
    """Load the deep learning model saved by main.py"""
    model_path = 'botnet_detection_model.h5'
    
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            st.success("✅ Pre-trained deep learning model loaded from main.py!")
            return model, "deep_learning"
        except Exception as e:
            st.warning(f"⚠️ Could not load pre-trained model: {e}")
    
    return None, None

# Train scikit-learn model as fallback
@st.cache_resource
def train_fallback_model(df):
    """Train a scikit-learn model as fallback when TensorFlow model isn't available"""
    try:
        if df is not None and 'is_botnet' in df.columns:
            X = df[['packet_size', 'interval']].values
            y = df['is_botnet'].values
            
            # Split data using same approach as main.py
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features (same as main.py)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            model.fit(X_train_scaled, y_train)
            
            # Calculate accuracy
            train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
            test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
            
            st.info(f"📊 Fallback model trained - Training Accuracy: {train_accuracy:.3f} | Test Accuracy: {test_accuracy:.3f}")
            
            return model, scaler, test_accuracy, "random_forest"
        
        else:
            st.error("❌ Unable to train model: Invalid dataset")
            return None, None, 0.0, None
            
    except Exception as e:
        st.error(f"❌ Error training model: {e}")
        return None, None, 0.0, None

# Load data using the actual path from main.py
df = load_data()

if df is not None:
    # Try to load the pre-trained deep learning model first
    pretrained_model, model_type = load_pretrained_model()
    
    if pretrained_model is not None:
        # Use the pre-trained deep learning model
        model = pretrained_model
        scaler = StandardScaler()  # We'll need to fit this on the data
        X = df[['packet_size', 'interval']].values
        scaler.fit(X)
        model_accuracy = 0.95  # Placeholder - you could calculate this if you have test data
        model_type_display = "Deep Learning (from main.py)"
    else:
        # Fall back to training a new scikit-learn model
        model, scaler, model_accuracy, model_type = train_fallback_model(df)
        model_type_display = "Random Forest (fallback)"

    if model is not None and scaler is not None:
        # Display basic info about the dataset
        st.subheader("📊 Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            botnet_count = df['is_botnet'].sum()
            st.metric("Botnet Records", int(botnet_count))
        
        with col3:
            normal_count = len(df) - df['is_botnet'].sum()
            st.metric("Normal Records", int(normal_count))
        
        with col4:
            st.metric("Model Type", model_type_display.split('(')[0])
        
        # Display data statistics matching main.py output
        st.subheader("📈 Data Statistics (matching main.py analysis)")
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.write("**Feature Statistics:**")
            feature_stats = df[['packet_size', 'interval']].describe()
            st.dataframe(feature_stats)
        
        with stats_col2:
            st.write("**Class Distribution:**")
            class_dist = df['is_botnet'].value_counts()
            st.write(f"- Normal (0): {class_dist[0]:,} samples")
            st.write(f"- Botnet (1): {class_dist[1]:,} samples")
            st.write(f"- Balance ratio: {class_dist[0]/class_dist[1]:.2f}:1")
        
        # Dataset preview
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(10))
        
        # User input for prediction
        st.subheader("🔍 Predict Botnet Activity")
        st.write(f"**Using:** {model_type_display}")
        
        # Get ranges from actual data
        min_packet = float(df['packet_size'].min())
        max_packet = float(df['packet_size'].max())
        mean_packet = float(df['packet_size'].mean())
        
        min_interval = float(df['interval'].min())
        max_interval = float(df['interval'].max())
        mean_interval = float(df['interval'].mean())
        
        col1, col2 = st.columns(2)
        with col1:
            packet_size = st.number_input(
                f"Packet Size (Range: {min_packet:.1f} - {max_packet:.1f})", 
                min_value=0.0, 
                max_value=max_packet * 2,
                value=mean_packet,
                step=10.0,
                help="Size of the network packet in bytes"
            )
        
        with col2:
            interval = st.number_input(
                f"Interval (Range: {min_interval:.2f} - {max_interval:.2f})", 
                min_value=0.0, 
                max_value=max_interval * 2,
                value=mean_interval,
                step=0.1,
                help="Time interval between packets in seconds"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔮 Predict", type="primary", use_container_width=True):
                try:
                    # Prepare input data
                    input_data = np.array([[packet_size, interval]])
                    input_scaled = scaler.transform(input_data)
                    
                    # Get prediction based on model type
                    if model_type == "deep_learning":
                        # Deep learning model prediction
                        prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
                        prediction = 1 if prediction_prob > 0.5 else 0
                        confidence = prediction_prob if prediction == 1 else (1 - prediction_prob)
                        probability = [1 - prediction_prob, prediction_prob]
                    else:
                        # Scikit-learn model prediction
                        prediction = model.predict(input_scaled)[0]
                        probability = model.predict_proba(input_scaled)[0]
                        confidence = max(probability)
                    
                    # Display result with styling
                    if prediction == 1:
                        st.error(f"🔴 **Botnet Activity Detected!**")
                        st.error(f"Confidence: {confidence:.1%}")
                    else:
                        st.success(f"🟢 **Normal Activity**")
                        st.success(f"Confidence: {confidence:.1%}")
                    
                    # Show additional details
                    with st.expander("📊 Prediction Details"):
                        st.write(f"**Prediction:** {'Botnet' if prediction == 1 else 'Normal'}")
                        st.write(f"**Normal Probability:** {probability[0]:.4f}")
                        st.write(f"**Botnet Probability:** {probability[1]:.4f}")
                        st.write(f"**Model Type:** {model_type_display}")
                        st.write(f"**Input values:** Packet Size = {packet_size:.2f}, Interval = {interval:.2f}")
                        
                        # Show how this compares to dataset
                        percentile_packet = (df['packet_size'] < packet_size).mean() * 100
                        percentile_interval = (df['interval'] < interval).mean() * 100
                        st.write(f"**Data percentiles:** Packet size = {percentile_packet:.1f}th, Interval = {percentile_interval:.1f}th")
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
        
        with col2:
            if st.button("🎲 Use Random Sample from Dataset", use_container_width=True):
                # Select a random sample from the actual dataset
                sample = df.sample(1).iloc[0]
                st.rerun()
        
        # Visualization section using actual data
        st.subheader("📊 Data Visualization (Actual Dataset)")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["🎯 Scatter Plot", "📈 Distributions", "⚖️ Class Balance"])
        
        with tab1:
            st.write("**Packet Size vs Interval Analysis (Real Data)**")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create scatter plot with actual data
            colors = ['#2E8B57', '#DC143C']  # Sea green and crimson
            labels = ['Normal', 'Botnet']
            
            for i, (label, color) in enumerate(zip(labels, colors)):
                mask = df['is_botnet'] == i
                if mask.any():
                    ax.scatter(df[mask]['packet_size'], df[mask]['interval'], 
                             alpha=0.6, s=50, c=color, label=label, edgecolors='white', linewidth=0.5)
            
            ax.set_xlabel("Packet Size (bytes)", fontsize=12)
            ax.set_ylabel("Interval (seconds)", fontsize=12)
            ax.set_title("Network Traffic Patterns: Botnet vs Normal Activity (Real Data)", fontsize=14, fontweight='bold')
            ax.legend(title='Activity Type', title_fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Add interpretation
            st.info("""
            **Interpretation Guide (Real Data):**
            - 🟢 **Green points (Normal):** Actual normal network behavior from your dataset
            - 🔴 **Red points (Botnet):** Actual botnet activity patterns from your dataset
            - These patterns reflect the real data distributions used in main.py training
            """)
        
        with tab2:
            st.write("**Feature Distribution Analysis (Real Data)**")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Packet size distribution
            for class_val, color, label in zip([0, 1], ['green', 'red'], ['Normal', 'Botnet']):
                subset = df[df['is_botnet'] == class_val]
                if len(subset) > 0:
                    axes[0, 0].hist(subset['packet_size'], alpha=0.7, color=color,
                                   label=label, bins=30, edgecolor='black', linewidth=0.5)
            axes[0, 0].set_xlabel("Packet Size (bytes)")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("Packet Size Distribution (Real Data)")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Interval distribution
            for class_val, color, label in zip([0, 1], ['green', 'red'], ['Normal', 'Botnet']):
                subset = df[df['is_botnet'] == class_val]
                if len(subset) > 0:
                    axes[0, 1].hist(subset['interval'], alpha=0.7, color=color,
                                   label=label, bins=30, edgecolor='black', linewidth=0.5)
            axes[0, 1].set_xlabel("Interval (seconds)")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Interval Distribution (Real Data)")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Box plots
            try:
                df_melted = df.melt(id_vars=['is_botnet'], value_vars=['packet_size'], 
                                   var_name='feature', value_name='value')
                sns.boxplot(data=df_melted, x='is_botnet', y='value', ax=axes[1, 0])
                axes[1, 0].set_xlabel("Activity Type (0=Normal, 1=Botnet)")
                axes[1, 0].set_ylabel("Packet Size")
                axes[1, 0].set_title("Packet Size by Activity Type")
                
                df_melted2 = df.melt(id_vars=['is_botnet'], value_vars=['interval'], 
                                    var_name='feature', value_name='value')
                sns.boxplot(data=df_melted2, x='is_botnet', y='value', ax=axes[1, 1])
                axes[1, 1].set_xlabel("Activity Type (0=Normal, 1=Botnet)")
                axes[1, 1].set_ylabel("Interval")
                axes[1, 1].set_title("Interval by Activity Type")
            except Exception as e:
                st.warning(f"Could not create box plots: {e}")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            st.write("**Class Distribution Analysis (Real Data)**")
            
            class_counts = df['is_botnet'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                colors = ['#98FB98', '#FFB6C1']  # Light green and light pink
                wedges, texts, autotexts = ax.pie(class_counts.values, labels=['Normal', 'Botnet'], 
                                                 autopct='%1.1f%%', colors=colors, startangle=90,
                                                 explode=(0.05, 0.05), shadow=True)
                
                # Enhance text
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
                
                ax.set_title("Class Distribution (Real Data)", fontsize=16, fontweight='bold', pad=20)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Enhanced bar chart
                fig, ax = plt.subplots(figsize=(8, 8))
                bars = ax.bar(['Normal', 'Botnet'], class_counts.values, 
                             color=['#98FB98', '#FFB6C1'], edgecolor='black', linewidth=1.5)
                ax.set_ylabel("Count", fontsize=12)
                ax.set_title("Activity Type Counts (Real Data)", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                plt.close()

else:
    st.error("❌ Unable to load dataset from main.py path")
    st.markdown("""
    ### 🔧 Setup Instructions:
    
    1. **Ensure your dataset exists** at one of these locations:
       - `/content/synthetic_robot_logs.csv` (main.py default)
       - `./synthetic_robot_logs.csv` (current directory)
       - `data/synthetic_robot_logs.csv` (data subfolder)
    
    2. **Run main.py first** to generate the dataset and train the deep learning model
    
    3. **Required columns** in your CSV:
       - `packet_size` (numeric)
       - `interval` (numeric) 
       - `is_botnet` (binary: 0 or 1)
    
    4. **Optional**: The app will automatically use the trained model from `botnet_detection_model.h5` if available
    """)

# Sidebar with enhanced information
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("""
This application uses the **actual data and models from main.py** for botnet detection in robotic network logs.

**Integration with main.py:**
- 📁 Uses same data path: `/content/synthetic_robot_logs.csv`
- 🤖 Loads pre-trained deep learning model if available
- 📊 Shows same statistics as main.py analysis
- 🎯 Provides interactive interface for the trained model
""")

st.sidebar.header("🚀 Model Priority")
st.sidebar.write("""
**Model Loading Order:**
1. 🥇 **Deep Learning Model** - From main.py (`botnet_detection_model.h5`)
2. 🥈 **Random Forest** - Fallback if DL model not found

**Current Status:**
""")

if df is not None:
    if 'model_type_display' in locals():
        if "Deep Learning" in model_type_display:
            st.sidebar.success("✅ Using Deep Learning model from main.py!")
        else:
            st.sidebar.info("ℹ️ Using fallback Random Forest model")
    st.sidebar.success(f"✅ Dataset loaded with {len(df):,} records")
else:
    st.sidebar.error("❌ No dataset loaded")

st.sidebar.header("📁 Expected File Structure")
st.sidebar.code("""
project/
├── main.py (your main script)
├── app.py (this streamlit app)
├── synthetic_robot_logs.csv
└── botnet_detection_model.h5 (generated)
""")

st.sidebar.header("🔧 Technical Details")
st.sidebar.write("""
**Data Processing:**
- Same StandardScaler as main.py
- Identical train/test split (80/20)
- Matching feature normalization

**Model Architecture:**
- Deep Learning: 4-layer neural network
- Fallback: Random Forest (100 trees)
- Both use same preprocessing pipeline
""")
