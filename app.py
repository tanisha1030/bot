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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import pickle
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="Botnet Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'df' not in st.session_state:
    st.session_state.df = None

def load_data(file_path):
    """Load the botnet detection dataset"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_model(input_shape):
    """Create the neural network model - exactly as in your original code"""
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
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def train_model_pipeline(df):
    """Complete training pipeline from your original code"""
    # Features and labels
    X = df[['packet_size', 'interval']].values
    y = df['is_botnet'].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stratified train-test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    # Compute class weights to handle imbalance
    weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights = {i: weights[i] for i in range(len(weights))}
    
    # Create model
    model = create_model(X_train.shape[1])
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )
    
    return model, scaler, history, X_test, y_test, class_weights

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    metrics = {
        'confusion_matrix': cm,
        'classification_report': classification_rep,
        'roc_auc_score': roc_auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'y_pred_probs': y_pred_probs,
        'y_pred': y_pred
    }
    
    return metrics

def plot_training_history(history):
    """Plot training history"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy',
                       'Training & Validation Precision', 'Training & Validation Recall'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Loss plot
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['loss'], 
                            name='Training Loss', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_loss'], 
                            name='Validation Loss', line=dict(color='red')), row=1, col=1)
    
    # Accuracy plot
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['accuracy'], 
                            name='Training Accuracy', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_accuracy'], 
                            name='Validation Accuracy', line=dict(color='orange')), row=1, col=2)
    
    # Precision plot
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['precision'], 
                            name='Training Precision', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_precision'], 
                            name='Validation Precision', line=dict(color='brown')), row=2, col=1)
    
    # Recall plot
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['recall'], 
                            name='Training Recall', line=dict(color='pink')), row=2, col=2)
    fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_recall'], 
                            name='Validation Recall', line=dict(color='gray')), row=2, col=2)
    
    fig.update_layout(height=600, title_text="Training History")
    fig.update_xaxes(title_text="Epoch")
    
    return fig

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Botnet'],
        y=['Actual Normal', 'Actual Botnet'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig

def plot_data_distribution(df):
    """Plot data distribution"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Packet Size Distribution', 'Interval Distribution',
                       'Packet Size vs Interval', 'Class Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    # Packet size distribution
    normal_data = df[df['is_botnet'] == 0]
    botnet_data = df[df['is_botnet'] == 1]
    
    fig.add_trace(go.Histogram(x=normal_data['packet_size'], name='Normal', 
                              opacity=0.7, nbinsx=50), row=1, col=1)
    fig.add_trace(go.Histogram(x=botnet_data['packet_size'], name='Botnet', 
                              opacity=0.7, nbinsx=50), row=1, col=1)
    
    # Interval distribution
    fig.add_trace(go.Histogram(x=normal_data['interval'], name='Normal', 
                              opacity=0.7, nbinsx=50, showlegend=False), row=1, col=2)
    fig.add_trace(go.Histogram(x=botnet_data['interval'], name='Botnet', 
                              opacity=0.7, nbinsx=50, showlegend=False), row=1, col=2)
    
    # Scatter plot
    fig.add_trace(go.Scatter(x=normal_data['packet_size'], y=normal_data['interval'], 
                            mode='markers', name='Normal', opacity=0.6, 
                            showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=botnet_data['packet_size'], y=botnet_data['interval'], 
                            mode='markers', name='Botnet', opacity=0.6, 
                            showlegend=False), row=2, col=1)
    
    # Pie chart
    class_counts = df['is_botnet'].value_counts()
    fig.add_trace(go.Pie(labels=['Normal', 'Botnet'], values=class_counts.values,
                        name="Class Distribution"), row=2, col=2)
    
    fig.update_layout(height=800, title_text="Data Analysis")
    
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">🔒 Botnet Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Home", "Data Analysis", "Model Training", "Model Evaluation", "Prediction"])
    
    if page == "Home":
        st.markdown("## Welcome to the Botnet Detection System")
        st.markdown("""
        This application uses deep learning to detect botnet traffic based on network packet characteristics.
        
        ### Features:
        - 📊 **Data Analysis**: Explore the dataset and visualize distributions
        - 🤖 **Model Training**: Train a deep neural network for botnet detection
        - 📈 **Model Evaluation**: Assess model performance with detailed metrics
        - 🔍 **Prediction**: Make predictions on new network traffic data
        
        ### Getting Started:
        1. Upload your `synthetic_robot_logs.csv` file
        2. Explore the data in the Data Analysis section
        3. Train the model in the Model Training section
        4. Evaluate results and make predictions
        """)
        
        # File upload
        st.markdown("### Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose your CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("✅ Dataset loaded successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dataset Shape", f"{df.shape[0]} × {df.shape[1]}")
            with col2:
                st.metric("Normal Traffic", f"{(df['is_botnet'] == 0).sum()}")
            with col3:
                st.metric("Botnet Traffic", f"{(df['is_botnet'] == 1).sum()}")
                
            st.markdown("### Sample Data")
            st.dataframe(df.head())
        
        elif st.session_state.df is None:
            st.warning("⚠️ Please upload your dataset to continue")
    
    elif page == "Data Analysis":
        st.markdown("## 📊 Data Analysis")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Basic statistics
            st.markdown("### Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", f"{df.shape[0]:,}")
            with col2:
                st.metric("Features", df.shape[1] - 1)
            with col3:
                st.metric("Normal Traffic", f"{(df['is_botnet'] == 0).sum():,}")
            with col4:
                st.metric("Botnet Traffic", f"{(df['is_botnet'] == 1).sum():,}")
            
            # Data distribution plot
            st.markdown("### Data Distribution")
            fig = plot_data_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.markdown("### Statistical Summary")
            st.dataframe(df.describe())
            
            # Missing values check
            st.markdown("### Data Quality")
            missing_values = df.isnull().sum().sum()
            if missing_values == 0:
                st.success("✅ No missing values found")
            else:
                st.warning(f"⚠️ Found {missing_values} missing values")
                
        else:
            st.warning("⚠️ Please upload your dataset first")
    
    elif page == "Model Training":
        st.markdown("## 🤖 Model Training")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            st.markdown("### Model Architecture")
            st.code("""
            Sequential([
                Input(shape=(2,)),
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
            """)
            
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training model... This may take a few minutes."):
                    try:
                        model, scaler, history, X_test, y_test, class_weights = train_model_pipeline(df)
                        
                        # Store in session state
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.training_history = history
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        
                        # Evaluate model
                        metrics = evaluate_model(model, X_test, y_test)
                        st.session_state.model_metrics = metrics
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Display training results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Training Epochs", len(history.history['loss']))
                        with col2:
                            st.metric("Final Loss", f"{history.history['loss'][-1]:.4f}")
                        with col3:
                            st.metric("Final Accuracy", f"{history.history['accuracy'][-1]:.4f}")
                        with col4:
                            st.metric("Class Weights", f"0: {class_weights[0]:.2f}, 1: {class_weights[1]:.2f}")
                        
                        # Plot training history
                        st.markdown("### Training History")
                        fig = plot_training_history(history)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
            
            if st.session_state.model is not None:
                st.markdown("### Model Information")
                st.success("✅ Model is trained and ready!")
                
                # Model summary
                if st.button("Show Model Summary"):
                    # Capture model summary
                    stringlist = []
                    st.session_state.model.summary(print_fn=lambda x: stringlist.append(x))
                    model_summary = "\n".join(stringlist)
                    st.code(model_summary)
                    
        else:
            st.warning("⚠️ Please upload your dataset first")
    
    elif page == "Model Evaluation":
        st.markdown("## 📈 Model Evaluation")
        
        if st.session_state.model is not None and st.session_state.model_metrics is not None:
            metrics = st.session_state.model_metrics
            
            # Performance metrics
            st.markdown("### Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['classification_report']['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            
            # ROC-AUC Score
            st.markdown("### ROC-AUC Score")
            st.metric("ROC-AUC", f"{metrics['roc_auc_score']:.4f}")
            
            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            fig_cm = plot_confusion_matrix(metrics['confusion_matrix'])
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Report
            st.markdown("### Detailed Classification Report")
            df_report = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(df_report)
            
            # Training History
            if st.session_state.training_history is not None:
                st.markdown("### Training History")
                fig_history = plot_training_history(st.session_state.training_history)
                st.plotly_chart(fig_history, use_container_width=True)
                
        else:
            st.warning("⚠️ Please train the model first")
    
    elif page == "Prediction":
        st.markdown("## 🔍 Prediction")
        
        if st.session_state.model is not None and st.session_state.scaler is not None:
            st.markdown("### Make Predictions")
            
            # Input form
            col1, col2 = st.columns(2)
            
            with col1:
                packet_size = st.number_input("Packet Size", 
                                            min_value=0.0, 
                                            max_value=10000.0, 
                                            value=1500.0, 
                                            step=1.0)
            
            with col2:
                interval = st.number_input("Interval", 
                                         min_value=0.0, 
                                         max_value=1.0, 
                                         value=0.1, 
                                         step=0.001, 
                                         format="%.3f")
            
            if st.button("🔍 Predict", type="primary"):
                # Prepare input
                input_data = np.array([[packet_size, interval]])
                input_scaled = st.session_state.scaler.transform(input_data)
                
                # Make prediction
                prediction_prob = st.session_state.model.predict(input_scaled, verbose=0)[0][0]
                prediction = 1 if prediction_prob > 0.5 else 0
                
                # Display result
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown('<div class="danger-box">⚠️ <strong>BOTNET DETECTED</strong></div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ <strong>NORMAL TRAFFIC</strong></div>', 
                                  unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence", f"{prediction_prob:.4f}")
                    st.metric("Prediction", "Botnet" if prediction == 1 else "Normal")
            
            # Batch prediction
            st.markdown("### Batch Prediction")
            st.markdown("Upload a CSV file with 'packet_size' and 'interval' columns for batch prediction")
            
            uploaded_file = st.file_uploader("Choose CSV file for batch prediction", type="csv")
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    
                    if 'packet_size' in batch_df.columns and 'interval' in batch_df.columns:
                        # Prepare data
                        X_batch = batch_df[['packet_size', 'interval']].values
                        X_batch_scaled = st.session_state.scaler.transform(X_batch)
                        
                        # Make predictions
                        predictions_prob = st.session_state.model.predict(X_batch_scaled, verbose=0)
                        predictions = (predictions_prob > 0.5).astype(int)
                        
                        # Add predictions to dataframe
                        batch_df['botnet_probability'] = predictions_prob
                        batch_df['predicted_class'] = predictions
                        batch_df['predicted_label'] = batch_df['predicted_class'].map({0: 'Normal', 1: 'Botnet'})
                        
                        st.markdown("### Batch Prediction Results")
                        st.dataframe(batch_df)
                        
                        # Summary
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Predictions", len(batch_df))
                        with col2:
                            st.metric("Botnet Detected", (batch_df['predicted_class'] == 1).sum())
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"botnet_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    else:
                        st.error("❌ CSV file must contain 'packet_size' and 'interval' columns")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    
        else:
            st.warning("⚠️ Please train the model first")

if __name__ == "__main__":
    main()
