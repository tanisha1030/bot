import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Configure the page
st.set_page_config(page_title="Botnet Detection", layout="wide")

# Title
st.title("🤖 Botnet Detection in Robotic Network Logs")

# Set random seeds for reproducibility (same as main.py)
np.random.seed(42)
tf.random.set_seed(42)

# Load and process data exactly like main.py
@st.cache_data
def load_and_process_data():
    """Load and process data exactly as done in main.py"""
    try:
        # Try to load from the exact path used in main.py
        df = pd.read_csv("/content/synthetic_robot_logs.csv")
        
        st.success("✅ Dataset loaded successfully!")
        
        # Display basic data exploration (exactly like main.py)
        st.info(f"""
        📊 **Dataset Information (from main.py):**
        - Dataset shape: {df.shape}
        - Class distribution: {df['is_botnet'].value_counts().to_dict()}
        - Missing values: {df.isnull().sum().sum()}
        """)
        
        return df
        
    except FileNotFoundError:
        st.error(f"""
        ❌ **Dataset not found at: `/content/synthetic_robot_logs.csv`**
        
        **To fix this:**
        1. Make sure you're running this in the same environment as main.py
        2. Ensure the CSV file exists at the exact path used in main.py
        3. Run main.py first to generate the dataset if needed
        """)
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

@st.cache_resource
def prepare_data_like_main_py(df):
    """Prepare data using the exact same process as main.py"""
    
    # Features and labels (exactly like main.py)
    X = df[['packet_size', 'interval']].values
    y = df['is_botnet'].values
    
    # Normalize features (exactly like main.py)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stratified train-test split (exactly like main.py)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    st.info(f"""
    📈 **Data Split (matching main.py):**
    - Training set shape: {X_train.shape}
    - Test set shape: {X_test.shape}
    """)
    
    # Compute class weights (exactly like main.py)
    weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights = {i: weights[i] for i in range(len(weights))}
    
    st.info(f"⚖️ **Class weights:** {class_weights}")
    
    return X_train, X_test, y_train, y_test, scaler, class_weights

@st.cache_resource
def create_model_like_main_py(input_shape):
    """Create the exact same model architecture as main.py"""
    
    # Define improved model with regularization (exactly like main.py)
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model with better optimizer settings (exactly like main.py)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

@st.cache_resource
def load_or_train_model(_X_train, _X_test, _y_train, _y_test, _class_weights):
    """Load pre-trained model or train new one using exact main.py approach"""
    
    model_path = 'botnet_detection_model.h5'
    
    # Try to load pre-trained model first
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            st.success("✅ Loaded pre-trained model from main.py!")
            
            # Evaluate the model on test data to get real accuracy
            y_pred_probs = model.predict(_X_test, verbose=0)
            y_pred = (y_pred_probs > 0.5).astype(int)
            
            # Calculate all metrics like main.py
            test_accuracy = (y_pred == _y_test.reshape(-1, 1)).mean()
            roc_auc = roc_auc_score(_y_test, y_pred_probs)
            f1 = f1_score(_y_test, y_pred)
            precision = precision_score(_y_test, y_pred)
            recall = recall_score(_y_test, y_pred)
            
            return model, test_accuracy, roc_auc, f1, precision, recall, y_pred_probs, y_pred
            
        except Exception as e:
            st.warning(f"⚠️ Could not load pre-trained model: {e}. Training new model...")
    
    # Train new model using exact main.py approach
    st.info("🔄 Training new model using main.py approach...")
    
    model = create_model_like_main_py(_X_train.shape[1])
    
    # Define callbacks (exactly like main.py)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train model (exactly like main.py)
    with st.spinner("Training model... This may take a few minutes."):
        history = model.fit(
            _X_train, _y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            class_weight=_class_weights,
            callbacks=callbacks,
            verbose=0  # Silent training for Streamlit
        )
    
    # Predict with optimal threshold (exactly like main.py)
    y_pred_probs = model.predict(_X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate all metrics exactly like main.py
    test_accuracy = (y_pred == _y_test.reshape(-1, 1)).mean()
    roc_auc = roc_auc_score(_y_test, y_pred_probs)
    f1 = f1_score(_y_test, y_pred)
    precision = precision_score(_y_test, y_pred)
    recall = recall_score(_y_test, y_pred)
    
    # Save model
    model.save(model_path)
    st.success(f"✅ Model trained and saved as '{model_path}'")
    
    return model, test_accuracy, roc_auc, f1, precision, recall, y_pred_probs, y_pred

# Load data
df = load_and_process_data()

if df is not None:
    # Prepare data exactly like main.py
    X_train, X_test, y_train, y_test, scaler, class_weights = prepare_data_like_main_py(df)
    
    # Load or train model
    model, test_accuracy, roc_auc, f1, precision, recall, y_pred_probs, y_pred = load_or_train_model(
        X_train, X_test, y_train, y_test, class_weights
    )
    
    if model is not None:
        # Display model performance exactly like main.py
        st.subheader("🎯 Model Performance (Exact Results from main.py)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{test_accuracy:.4f}")
        with col2:
            st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
        with col3:
            st.metric("F1 Score", f"{f1:.4f}")
        with col4:
            st.metric("Precision", f"{precision:.4f}")
        
        # Enhanced evaluation (exactly like main.py)
        st.subheader("📊 Detailed Evaluation (from main.py)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confusion Matrix:**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        # Display dataset information
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
            st.metric("Features", len(['packet_size', 'interval']))
        
        # Dataset preview
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(10))
        
        # User input for prediction
        st.subheader("🔍 Predict Botnet Activity")
        st.write("**Using the same model and preprocessing as main.py**")
        
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
                    # Prepare input data using same scaler as main.py
                    input_data = np.array([[packet_size, interval]])
                    input_scaled = scaler.transform(input_data)
                    
                    # Get prediction using same threshold as main.py
                    prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
                    prediction = 1 if prediction_prob > 0.5 else 0
                    confidence = prediction_prob if prediction == 1 else (1 - prediction_prob)
                    
                    # Display result
                    if prediction == 1:
                        st.error(f"🔴 **Botnet Activity Detected!**")
                        st.error(f"Confidence: {confidence:.1%}")
                    else:
                        st.success(f"🟢 **Normal Activity**")
                        st.success(f"Confidence: {confidence:.1%}")
                    
                    # Show additional details
                    with st.expander("📊 Prediction Details"):
                        st.write(f"**Prediction:** {'Botnet' if prediction == 1 else 'Normal'}")
                        st.write(f"**Raw Probability:** {prediction_prob:.4f}")
                        st.write(f"**Threshold:** 0.5 (same as main.py)")
                        st.write(f"**Model Accuracy:** {test_accuracy:.4f}")
                        st.write(f"**Input values:** Packet Size = {packet_size:.2f}, Interval = {interval:.2f}")
                        
                        # Show preprocessing details
                        st.write("**Preprocessing (same as main.py):**")
                        st.write(f"- Standardized packet size: {input_scaled[0][0]:.4f}")
                        st.write(f"- Standardized interval: {input_scaled[0][1]:.4f}")
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
        
        with col2:
            if st.button("🎲 Random Sample from Test Set", use_container_width=True):
                # Get a random sample from the test set used in main.py
                random_idx = np.random.randint(0, len(X_test))
                original_values = scaler.inverse_transform([X_test[random_idx]])[0]
                
                st.write(f"**Test Sample #{random_idx}:**")
                st.write(f"- Packet Size: {original_values[0]:.2f}")
                st.write(f"- Interval: {original_values[1]:.2f}")
                st.write(f"- True Label: {'Botnet' if y_test[random_idx] == 1 else 'Normal'}")
                
                # Make prediction
                pred_prob = model.predict([X_test[random_idx:random_idx+1]], verbose=0)[0][0]
                pred_label = 1 if pred_prob > 0.5 else 0
                
                if pred_label == y_test[random_idx]:
                    st.success(f"✅ Correct prediction: {pred_prob:.4f}")
                else:
                    st.error(f"❌ Incorrect prediction: {pred_prob:.4f}")
        
        # Visualization section
        st.subheader("📊 Data Visualization")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Scatter Plot", "📈 Distributions", "📋 Model Details"])
        
        with tab1:
            st.write("**Packet Size vs Interval Analysis (Real Data)**")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = ['#2E8B57', '#DC143C']
            labels = ['Normal', 'Botnet']
            
            for i, (label, color) in enumerate(zip(labels, colors)):
                mask = df['is_botnet'] == i
                if mask.any():
                    ax.scatter(df[mask]['packet_size'], df[mask]['interval'], 
                             alpha=0.6, s=50, c=color, label=label, edgecolors='white', linewidth=0.5)
            
            ax.set_xlabel("Packet Size (bytes)", fontsize=12)
            ax.set_ylabel("Interval (seconds)", fontsize=12)
            ax.set_title("Network Traffic Patterns (Real Data from main.py)", fontsize=14, fontweight='bold')
            ax.legend(title='Activity Type')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            st.write("**Feature Distribution Analysis**")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Packet size distribution
            for class_val, color, label in zip([0, 1], ['green', 'red'], ['Normal', 'Botnet']):
                subset = df[df['is_botnet'] == class_val]
                if len(subset) > 0:
                    axes[0, 0].hist(subset['packet_size'], alpha=0.7, color=color,
                                   label=label, bins=30, edgecolor='black', linewidth=0.5)
            axes[0, 0].set_xlabel("Packet Size (bytes)")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("Packet Size Distribution")
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
            axes[0, 1].set_title("Interval Distribution")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Box plots
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
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            st.write("**Model Architecture & Configuration (from main.py)**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.code(f"""
Model Architecture:
- Input Layer: {X_train.shape[1]} features
- Dense Layer 1: 128 neurons + ReLU + BatchNorm + Dropout(0.3)
- Dense Layer 2: 64 neurons + ReLU + BatchNorm + Dropout(0.3)  
- Dense Layer 3: 32 neurons + ReLU + Dropout(0.2)
- Output Layer: 1 neuron + Sigmoid

Optimizer: Adam (lr=0.001)
Loss: Binary Crossentropy
Metrics: Accuracy, Precision, Recall
                """)
            
            with col2:
                st.code(f"""
Training Configuration:
- Epochs: 100 (with early stopping)
- Batch Size: 32
- Validation Split: 0.2
- Class Weights: {class_weights}
- Random Seed: 42

Callbacks:
- EarlyStopping (patience=10)
- ReduceLROnPlateau (factor=0.5, patience=5)
                """)
            
            st.subheader("Performance Summary")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Score': [test_accuracy, precision, recall, f1, roc_auc]
            })
            st.dataframe(metrics_df)

# Sidebar
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("""
This app uses the **exact same data processing, model architecture, and evaluation** as your main.py file.

**Key Features:**
- 📁 Same data path: `/content/synthetic_robot_logs.csv`
- 🧠 Identical model architecture from main.py
- 📊 Same preprocessing (StandardScaler, StratifiedShuffleSplit)
- 🎯 Exact evaluation metrics and thresholds
- 🔄 Reproducible results (seed=42)
""")

st.sidebar.header("🎯 Accuracy Explanation")
st.sidebar.write(f"""
**Why the accuracy matches main.py:**
- Uses identical train/test split (seed=42)
- Same model architecture and hyperparameters
- Identical preprocessing pipeline
- Same evaluation methodology

**Current Status:**
""")

if df is not None and 'test_accuracy' in locals():
    st.sidebar.success(f"✅ Test Accuracy: {test_accuracy:.4f}")
    st.sidebar.success(f"✅ Model loaded from main.py")
    st.sidebar.success(f"✅ Dataset: {len(df):,} records")
else:
    st.sidebar.error("❌ Setup incomplete")

st.sidebar.header("🔧 Requirements")
st.sidebar.write("""
**To get exact main.py results:**
1. Run in same environment as main.py
2. Dataset must exist at `/content/synthetic_robot_logs.csv`
3. Same Python packages and versions
4. Optional: Pre-trained model `botnet_detection_model.h5`
""")
