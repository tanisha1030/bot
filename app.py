import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import time
import io
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🤖 Botnet Detection Simulation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .robot-status {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class BotnetRobotSimulator:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        
    def generate_synthetic_robot_data(self, n_samples=10000, botnet_ratio=0.15):
        """Generate synthetic robot network data simulating botnet behavior"""
        np.random.seed(42)
        
        normal_samples = int(n_samples * (1 - botnet_ratio))
        botnet_samples = n_samples - normal_samples
        
        # Normal robot behavior - legitimate automated tasks
        normal_packet_sizes = np.random.normal(512, 128, normal_samples)
        normal_intervals = np.random.exponential(2.0, normal_samples)  # Slower, more predictable
        
        # Botnet robot behavior - malicious coordinated activity
        botnet_packet_sizes = np.random.normal(1024, 256, botnet_samples)
        botnet_intervals = np.random.exponential(0.5, botnet_samples)  # Faster, more aggressive
        
        # Add some noise and edge cases
        normal_packet_sizes = np.clip(normal_packet_sizes, 64, 1500)
        normal_intervals = np.clip(normal_intervals, 0.1, 10.0)
        botnet_packet_sizes = np.clip(botnet_packet_sizes, 64, 1500)
        botnet_intervals = np.clip(botnet_intervals, 0.01, 5.0)
        
        # Combine data
        packet_sizes = np.concatenate([normal_packet_sizes, botnet_packet_sizes])
        intervals = np.concatenate([normal_intervals, botnet_intervals])
        labels = np.concatenate([np.zeros(normal_samples), np.ones(botnet_samples)])
        
        # Create DataFrame
        df = pd.DataFrame({
            'packet_size': packet_sizes,
            'interval': intervals,
            'is_botnet': labels
        })
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def create_enhanced_model(self, input_shape):
        """Create the enhanced neural network model based on your original code"""
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
    
    def train_model(self, X_train, y_train, progress_bar=None):
        """Train the model with callbacks and progress tracking"""
        # Compute class weights
        weights = class_weight.compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weights = {i: weights[i] for i in range(len(weights))}
        
        # Create model
        self.model = self.create_enhanced_model(X_train.shape[1])
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Custom callback for Streamlit progress
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar, total_epochs):
                self.progress_bar = progress_bar
                self.total_epochs = total_epochs
                
            def on_epoch_end(self, epoch, logs=None):
                if self.progress_bar:
                    self.progress_bar.progress((epoch + 1) / self.total_epochs)
        
        if progress_bar:
            callbacks.append(StreamlitCallback(progress_bar, 100))
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        
        return self.model, self.history

# Initialize the simulator
@st.cache_resource
def load_simulator():
    return BotnetRobotSimulator()

simulator = load_simulator()

# Main app
st.markdown('<div class="main-header">🤖 Botnet Detection Simulation</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("---")

# Data generation parameters
st.sidebar.subheader("Data Generation")
n_samples = st.sidebar.slider("Number of Samples", 1000, 50000, 10000, 1000)
botnet_ratio = st.sidebar.slider("Botnet Ratio", 0.05, 0.5, 0.15, 0.05)

# Model parameters
st.sidebar.subheader("Model Configuration")
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 100, 10)

# Generate data button
if st.sidebar.button("🔄 Generate New Data", type="primary"):
    st.session_state.data_generated = False

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Generation", "🧠 Model Training", "📈 Results", "🔍 Real-time Detection"])

with tab1:
    st.header("Robot Network Data Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Synthetic Robot Log Generation")
        
        if st.button("Generate Robot Network Data", type="primary") or 'df' not in st.session_state:
            with st.spinner("Generating synthetic robot network data..."):
                df = simulator.generate_synthetic_robot_data(n_samples, botnet_ratio)
                st.session_state.df = df
                st.session_state.data_generated = True
                
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Display basic statistics
            st.success(f"✅ Generated {len(df)} robot network samples")
            
            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                st.metric("Total Samples", len(df))
            with col1_2:
                normal_count = (df['is_botnet'] == 0).sum()
                st.metric("Normal Robots", normal_count)
            with col1_3:
                botnet_count = (df['is_botnet'] == 1).sum()
                st.metric("Botnet Robots", botnet_count)
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="synthetic_robot_logs.csv",
                mime="text/csv"
            )
    
    with col2:
        if 'df' in st.session_state:
            df = st.session_state.df
            
            # Class distribution
            fig_dist = px.pie(
                values=df['is_botnet'].value_counts().values,
                names=['Normal', 'Botnet'],
                title="Robot Type Distribution",
                color_discrete_map={'Normal': '#2E86AB', 'Botnet': '#F24236'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Feature distributions
            fig_features = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Packet Size Distribution', 'Interval Distribution'),
                vertical_spacing=0.1
            )
            
            for label, name, color in [(0, 'Normal', '#2E86AB'), (1, 'Botnet', '#F24236')]:
                data = df[df['is_botnet'] == label]
                
                fig_features.add_trace(
                    go.Histogram(x=data['packet_size'], name=f'{name} Packets', 
                               marker_color=color, opacity=0.7),
                    row=1, col=1
                )
                
                fig_features.add_trace(
                    go.Histogram(x=data['interval'], name=f'{name} Intervals', 
                               marker_color=color, opacity=0.7),
                    row=2, col=1
                )
            
            fig_features.update_layout(height=500, title_text="Feature Distributions")
            st.plotly_chart(fig_features, use_container_width=True)

with tab2:
    st.header("Deep Learning Model Training")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Please generate data first in the Data Generation tab.")
    else:
        df = st.session_state.df
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Train Model", type="primary"):
                # Prepare data (following your original code structure)
                X = df[['packet_size', 'interval']].values
                y = df['is_botnet'].values
                
                # Normalize features
                simulator.scaler = StandardScaler()
                X_scaled = simulator.scaler.fit_transform(X)
                
                # Stratified train-test split
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                for train_idx, test_idx in sss.split(X_scaled, y):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                
                # Store test data for later use
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_scaled = X_scaled
                
                # Display training info
                st.info(f"Training set shape: {X_train.shape}")
                st.info(f"Test set shape: {X_test.shape}")
                
                # Training with progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Training deep learning model..."):
                    status_text.text("Training in progress...")
                    model, history = simulator.train_model(X_train, y_train, progress_bar)
                    
                    # Store model and history
                    st.session_state.model = model
                    st.session_state.history = history
                    st.session_state.model_trained = True
                    
                status_text.text("Training completed!")
                st.success("✅ Model training completed successfully!")
                
                # Show training summary
                st.subheader("Model Architecture")
                buffer = io.StringIO()
                model.summary(print_fn=lambda x: buffer.write(x + '\n'))
                st.text(buffer.getvalue())
        
        with col2:
            if 'history' in st.session_state:
                st.subheader("Training Progress")
                
                # Plot training history
                history = st.session_state.history
                
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.history['loss'], 
                    name='Training Loss',
                    line=dict(color='#FF6B6B')
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history.history['val_loss'], 
                    name='Validation Loss',
                    line=dict(color='#4ECDC4')
                ))
                fig_loss.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig_loss, use_container_width=True)
                
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    y=history.history['accuracy'], 
                    name='Training Accuracy',
                    line=dict(color='#FF6B6B')
                ))
                fig_acc.add_trace(go.Scatter(
                    y=history.history['val_accuracy'], 
                    name='Validation Accuracy',
                    line=dict(color='#4ECDC4')
                ))
                fig_acc.update_layout(title="Training Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
                st.plotly_chart(fig_acc, use_container_width=True)

with tab3:
    st.header("Model Evaluation Results")
    
    if 'model_trained' not in st.session_state:
        st.warning("⚠️ Please train the model first in the Model Training tab.")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Make predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_probs)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
        with col2:
            st.metric("F1 Score", f"{f1:.4f}")
        with col3:
            st.metric("Precision", f"{precision:.4f}")
        with col4:
            st.metric("Recall", f"{recall:.4f}")
        
        # Confusion Matrix
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Confusion Matrix")
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                              labels=dict(x="Predicted", y="Actual"),
                              x=['Normal', 'Botnet'], y=['Normal', 'Botnet'])
            fig_cm.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {roc_auc:.4f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier'))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

with tab4:
    st.header("Real-time Botnet Detection")
    
    if 'model_trained' not in st.session_state:
        st.warning("⚠️ Please train the model first in the Model Training tab.")
    else:
        st.subheader("🔍 Test Individual Robot Behavior")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Enter Robot Network Parameters:**")
            packet_size = st.number_input("Packet Size (bytes)", min_value=64, max_value=1500, value=512)
            interval = st.number_input("Interval (seconds)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            
            if st.button("🔍 Detect Botnet", type="primary"):
                # Prepare input
                input_data = np.array([[packet_size, interval]])
                input_scaled = simulator.scaler.transform(input_data)
                
                # Make prediction
                prediction_prob = simulator.model.predict(input_scaled, verbose=0)[0][0]
                prediction = "🦠 BOTNET" if prediction_prob > 0.5 else "✅ NORMAL"
                
                # Display result
                with col2:
                    st.subheader("Detection Result")
                    
                    if prediction_prob > 0.5:
                        st.error(f"🚨 **BOTNET DETECTED** 🚨")
                        st.error(f"Confidence: {prediction_prob:.2%}")
                    else:
                        st.success(f"✅ **NORMAL ROBOT** ✅")
                        st.success(f"Confidence: {1-prediction_prob:.2%}")
                    
                    # Gauge chart for confidence
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Botnet Probability"},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 1], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Batch testing
        st.subheader("📊 Batch Testing")
        
        if st.button("🎲 Generate Random Test Batch"):
            # Generate random test samples
            test_samples = 100
            test_data = simulator.generate_synthetic_robot_data(test_samples, 0.3)
            
            X_batch = test_data[['packet_size', 'interval']].values
            y_batch = test_data['is_botnet'].values
            X_batch_scaled = simulator.scaler.transform(X_batch)
            
            # Make predictions
            predictions = simulator.model.predict(X_batch_scaled, verbose=0)
            predicted_labels = (predictions > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(predicted_labels.flatten() == y_batch)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Batch Size", test_samples)
            with col2:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col3:
                detected_botnets = np.sum(predicted_labels)
                st.metric("Detected Botnets", detected_botnets)
            
            # Show sample results
            results_df = pd.DataFrame({
                'Packet Size': X_batch[:10, 0],
                'Interval': X_batch[:10, 1],
                'Actual': ['Botnet' if x == 1 else 'Normal' for x in y_batch[:10]],
                'Predicted': ['Botnet' if x == 1 else 'Normal' for x in predicted_labels[:10].flatten()],
                'Confidence': [f"{x:.2%}" for x in predictions[:10].flatten()]
            })
            
            st.subheader("Sample Results")
            st.dataframe(results_df)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Botnet Detection Simulation | Built with Streamlit & TensorFlow</p>
    <p>Based on Deep Learning Neural Networks for Robot Network Analysis</p>
</div>
""", unsafe_allow_html=True)
