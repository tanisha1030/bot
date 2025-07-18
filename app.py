import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import time
import random
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Botnet Detection Robot Simulation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BotnetDetectionRobot:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ml_model = None
        self.dl_model = None
        self.detection_threshold = 0.5
        
    def generate_network_data(self, num_samples=1000):
        """Generate synthetic network traffic data"""
        np.random.seed(42)
        
        # Normal traffic features
        normal_traffic = {
            'packet_size': np.random.normal(512, 128, num_samples//2),
            'connection_duration': np.random.exponential(10, num_samples//2),
            'bytes_per_second': np.random.normal(1000, 200, num_samples//2),
            'unique_destinations': np.random.poisson(3, num_samples//2),
            'port_scan_attempts': np.random.poisson(0.1, num_samples//2),
            'failed_connections': np.random.poisson(0.5, num_samples//2),
            'dns_queries': np.random.poisson(5, num_samples//2),
            'protocol_anomalies': np.random.poisson(0.2, num_samples//2),
            'label': np.zeros(num_samples//2)
        }
        
        # Botnet traffic features (more suspicious patterns)
        botnet_traffic = {
            'packet_size': np.random.normal(256, 64, num_samples//2),
            'connection_duration': np.random.exponential(2, num_samples//2),
            'bytes_per_second': np.random.normal(2000, 500, num_samples//2),
            'unique_destinations': np.random.poisson(15, num_samples//2),
            'port_scan_attempts': np.random.poisson(3, num_samples//2),
            'failed_connections': np.random.poisson(5, num_samples//2),
            'dns_queries': np.random.poisson(20, num_samples//2),
            'protocol_anomalies': np.random.poisson(2, num_samples//2),
            'label': np.ones(num_samples//2)
        }
        
        # Combine data
        data = {}
        for key in normal_traffic.keys():
            data[key] = np.concatenate([normal_traffic[key], botnet_traffic[key]])
        
        df = pd.DataFrame(data)
        return df.sample(frac=1).reset_index(drop=True)
    
    def train_ml_model(self, X_train, y_train):
        """Train machine learning models"""
        # Random Forest for classification
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train, y_train)
        
        # Isolation Forest for anomaly detection
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_model.fit(X_train)
        
    def create_dl_model(self, input_shape):
        """Create deep learning model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_dl_model(self, X_train, y_train, X_val, y_val):
        """Train deep learning model"""
        self.dl_model = self.create_dl_model(X_train.shape[1])
        
        history = self.dl_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        return history
    
    def simulate_real_time_detection(self, data_sample):
        """Simulate real-time botnet detection"""
        if self.ml_model is None or self.dl_model is None:
            return None
        
        # ML prediction
        ml_prediction = self.ml_model.predict_proba(data_sample)[0][1]
        
        # DL prediction
        dl_prediction = self.dl_model.predict(data_sample, verbose=0)[0][0]
        
        # Anomaly detection
        anomaly_score = self.anomaly_model.decision_function(data_sample)[0]
        
        # Ensemble prediction
        ensemble_prediction = (ml_prediction + dl_prediction) / 2
        
        return {
            'ml_prediction': ml_prediction,
            'dl_prediction': dl_prediction,
            'anomaly_score': anomaly_score,
            'ensemble_prediction': ensemble_prediction,
            'is_botnet': ensemble_prediction > self.detection_threshold
        }

def main():
    st.title("🤖 Botnet Detection Robot Simulation")
    st.markdown("### AI-Powered Network Security Monitoring System")
    
    # Initialize robot
    if 'robot' not in st.session_state:
        st.session_state.robot = BotnetDetectionRobot()
        st.session_state.data_generated = False
        st.session_state.models_trained = False
    
    # Sidebar
    st.sidebar.header("Control Panel")
    
    # Data Generation
    st.sidebar.subheader("1. Data Generation")
    num_samples = st.sidebar.slider("Number of Samples", 500, 5000, 1000)
    
    if st.sidebar.button("Generate Network Data"):
        with st.spinner("Generating synthetic network data..."):
            st.session_state.data = st.session_state.robot.generate_network_data(num_samples)
            st.session_state.data_generated = True
        st.success("Data generated successfully!")
    
    # Model Training
    st.sidebar.subheader("2. Model Training")
    if st.sidebar.button("Train Models", disabled=not st.session_state.data_generated):
        with st.spinner("Training ML and DL models..."):
            data = st.session_state.data
            
            # Prepare features
            feature_cols = ['packet_size', 'connection_duration', 'bytes_per_second', 
                          'unique_destinations', 'port_scan_attempts', 'failed_connections',
                          'dns_queries', 'protocol_anomalies']
            
            X = data[feature_cols]
            y = data['label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = st.session_state.robot.scaler.fit_transform(X_train)
            X_val_scaled = st.session_state.robot.scaler.transform(X_val)
            X_test_scaled = st.session_state.robot.scaler.transform(X_test)
            
            # Train models
            st.session_state.robot.train_ml_model(X_train_scaled, y_train)
            history = st.session_state.robot.train_dl_model(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Store for evaluation
            st.session_state.X_test = X_test_scaled
            st.session_state.y_test = y_test
            st.session_state.history = history
            st.session_state.models_trained = True
            
        st.success("Models trained successfully!")
    
    # Real-time Detection
    st.sidebar.subheader("3. Real-time Detection")
    detection_speed = st.sidebar.slider("Detection Speed (seconds)", 0.5, 5.0, 1.0)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Data Overview
        if st.session_state.data_generated:
            st.subheader("📊 Network Traffic Data")
            
            # Display data statistics
            data = st.session_state.data
            
            tab1, tab2, tab3 = st.tabs(["Data Overview", "Feature Analysis", "Traffic Patterns"])
            
            with tab1:
                st.write("Dataset Shape:", data.shape)
                st.write("Botnet Traffic Ratio:", f"{data['label'].mean():.2%}")
                st.dataframe(data.head())
            
            with tab2:
                # Feature correlation heatmap
                feature_cols = ['packet_size', 'connection_duration', 'bytes_per_second', 
                              'unique_destinations', 'port_scan_attempts', 'failed_connections',
                              'dns_queries', 'protocol_anomalies']
                
                corr_matrix = data[feature_cols].corr()
                fig = px.imshow(corr_matrix, 
                               title="Feature Correlation Matrix",
                               color_continuous_scale="RdBu")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Traffic patterns visualization
                fig = px.box(data, x='label', y='bytes_per_second', 
                           title="Bytes per Second by Traffic Type")
                fig.update_xaxis(tickvals=[0, 1], ticktext=['Normal', 'Botnet'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Model Performance
        if st.session_state.models_trained:
            st.subheader("🎯 Model Performance")
            
            # Evaluate models
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # ML model predictions
            ml_predictions = st.session_state.robot.ml_model.predict(X_test)
            ml_proba = st.session_state.robot.ml_model.predict_proba(X_test)[:, 1]
            
            # DL model predictions
            dl_predictions = (st.session_state.robot.dl_model.predict(X_test, verbose=0) > 0.5).astype(int)
            dl_proba = st.session_state.robot.dl_model.predict(X_test, verbose=0).flatten()
            
            # Display metrics
            col1_metrics, col2_metrics = st.columns(2)
            
            with col1_metrics:
                st.write("**Random Forest Performance:**")
                ml_report = classification_report(y_test, ml_predictions, output_dict=True)
                st.write(f"Accuracy: {ml_report['accuracy']:.3f}")
                st.write(f"Precision: {ml_report['1']['precision']:.3f}")
                st.write(f"Recall: {ml_report['1']['recall']:.3f}")
                st.write(f"F1-Score: {ml_report['1']['f1-score']:.3f}")
            
            with col2_metrics:
                st.write("**Deep Learning Performance:**")
                dl_report = classification_report(y_test, dl_predictions, output_dict=True)
                st.write(f"Accuracy: {dl_report['accuracy']:.3f}")
                st.write(f"Precision: {dl_report['1']['precision']:.3f}")
                st.write(f"Recall: {dl_report['1']['recall']:.3f}")
                st.write(f"F1-Score: {dl_report['1']['f1-score']:.3f}")
            
            # Training history
            if 'history' in st.session_state:
                history = st.session_state.history
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Training Accuracy'))
                fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'))
                fig.update_layout(title="Deep Learning Model Training History", 
                                xaxis_title="Epoch", yaxis_title="Accuracy")
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Real-time Detection Panel
        st.subheader("🔍 Real-time Detection")
        
        if st.session_state.models_trained:
            # Detection controls
            auto_detect = st.checkbox("Auto Detection", value=False)
            
            if auto_detect:
                # Placeholder for real-time detection
                detection_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                # Simulate real-time detection
                detection_count = 0
                botnet_detected = 0
                
                while auto_detect:
                    # Generate random sample
                    sample_data = st.session_state.data.sample(1)
                    feature_cols = ['packet_size', 'connection_duration', 'bytes_per_second', 
                                  'unique_destinations', 'port_scan_attempts', 'failed_connections',
                                  'dns_queries', 'protocol_anomalies']
                    
                    sample_features = st.session_state.robot.scaler.transform(sample_data[feature_cols])
                    
                    # Get prediction
                    result = st.session_state.robot.simulate_real_time_detection(sample_features)
                    
                    detection_count += 1
                    if result['is_botnet']:
                        botnet_detected += 1
                    
                    # Update display
                    with detection_placeholder.container():
                        if result['is_botnet']:
                            st.error("⚠️ BOTNET DETECTED!")
                        else:
                            st.success("✅ Normal Traffic")
                        
                        st.write(f"**Predictions:**")
                        st.write(f"ML Score: {result['ml_prediction']:.3f}")
                        st.write(f"DL Score: {result['dl_prediction']:.3f}")
                        st.write(f"Ensemble: {result['ensemble_prediction']:.3f}")
                        st.write(f"Anomaly Score: {result['anomaly_score']:.3f}")
                    
                    with metrics_placeholder.container():
                        st.write(f"**Detection Stats:**")
                        st.write(f"Total Scans: {detection_count}")
                        st.write(f"Threats Detected: {botnet_detected}")
                        st.write(f"Detection Rate: {botnet_detected/detection_count:.2%}")
                    
                    time.sleep(detection_speed)
                    
                    # Break if auto_detect is unchecked
                    if not st.session_state.get('auto_detect', True):
                        break
            
            # Manual detection
            if st.button("🔍 Scan Single Sample"):
                sample_data = st.session_state.data.sample(1)
                feature_cols = ['packet_size', 'connection_duration', 'bytes_per_second', 
                              'unique_destinations', 'port_scan_attempts', 'failed_connections',
                              'dns_queries', 'protocol_anomalies']
                
                sample_features = st.session_state.robot.scaler.transform(sample_data[feature_cols])
                result = st.session_state.robot.simulate_real_time_detection(sample_features)
                
                if result['is_botnet']:
                    st.error("⚠️ BOTNET DETECTED!")
                else:
                    st.success("✅ Normal Traffic")
                
                st.write(f"**Predictions:**")
                st.write(f"ML Score: {result['ml_prediction']:.3f}")
                st.write(f"DL Score: {result['dl_prediction']:.3f}")
                st.write(f"Ensemble: {result['ensemble_prediction']:.3f}")
                st.write(f"Anomaly Score: {result['anomaly_score']:.3f}")
        
        else:
            st.info("Train models first to enable real-time detection")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🔧 Robot Simulation Features")
    st.markdown("""
    - **Machine Learning**: Random Forest + Isolation Forest
    - **Deep Learning**: Neural Network with dropout regularization
    - **Real-time Detection**: Ensemble prediction system
    - **Anomaly Detection**: Unsupervised learning for unknown threats
    - **Interactive Visualization**: Real-time monitoring dashboard
    """)

if __name__ == "__main__":
    main()
