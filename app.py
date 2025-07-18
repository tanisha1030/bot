import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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
        self.dl_model = None  # Will use logistic regression instead
        self.anomaly_model = None
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
    
    def train_ml_models(self, X_train, y_train):
        """Train machine learning models"""
        # Random Forest for classification
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train, y_train)
        
        # Logistic Regression as "Deep Learning" alternative
        self.dl_model = LogisticRegression(max_iter=1000, random_state=42)
        self.dl_model.fit(X_train, y_train)
        
        # Isolation Forest for anomaly detection
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_model.fit(X_train)
    
    def simulate_real_time_detection(self, data_sample):
        """Simulate real-time botnet detection"""
        if self.ml_model is None or self.dl_model is None:
            return None
        
        # ML prediction
        ml_prediction = self.ml_model.predict_proba(data_sample)[0][1]
        
        # "DL" prediction (Logistic Regression)
        dl_prediction = self.dl_model.predict_proba(data_sample)[0][1]
        
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
    st.info("📝 Note: This version uses Logistic Regression instead of Neural Networks for better compatibility.")
    
    # Initialize robot
    if 'robot' not in st.session_state:
        st.session_state.robot = BotnetDetectionRobot()
        st.session_state.data_generated = False
        st.session_state.models_trained = False
    
    # Sidebar
    st.sidebar.header("🎛️ Control Panel")
    
    # Data Generation
    st.sidebar.subheader("1. Data Generation")
    num_samples = st.sidebar.slider("Number of Samples", 500, 5000, 1000)
    
    if st.sidebar.button("🔄 Generate Network Data"):
        with st.spinner("Generating synthetic network data..."):
            st.session_state.data = st.session_state.robot.generate_network_data(num_samples)
            st.session_state.data_generated = True
        st.success("✅ Data generated successfully!")
    
    # Model Training
    st.sidebar.subheader("2. Model Training")
    if st.sidebar.button("🧠 Train Models", disabled=not st.session_state.data_generated):
        with st.spinner("Training ML models..."):
            data = st.session_state.data
            
            # Prepare features
            feature_cols = ['packet_size', 'connection_duration', 'bytes_per_second', 
                          'unique_destinations', 'port_scan_attempts', 'failed_connections',
                          'dns_queries', 'protocol_anomalies']
            
            X = data[feature_cols]
            y = data['label']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = st.session_state.robot.scaler.fit_transform(X_train)
            X_test_scaled = st.session_state.robot.scaler.transform(X_test)
            
            # Train models
            st.session_state.robot.train_ml_models(X_train_scaled, y_train)
            
            # Store for evaluation
            st.session_state.X_test = X_test_scaled
            st.session_state.y_test = y_test
            st.session_state.feature_cols = feature_cols
            st.session_state.models_trained = True
            
        st.success("✅ Models trained successfully!")
    
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
            
            tab1, tab2, tab3 = st.tabs(["📈 Data Overview", "🔍 Feature Analysis", "🌐 Traffic Patterns"])
            
            with tab1:
                col1_stats, col2_stats = st.columns(2)
                
                with col1_stats:
                    st.metric("Total Samples", data.shape[0])
                    st.metric("Features", data.shape[1] - 1)
                
                with col2_stats:
                    botnet_ratio = data['label'].mean()
                    st.metric("Botnet Traffic", f"{botnet_ratio:.2%}")
                    st.metric("Normal Traffic", f"{1-botnet_ratio:.2%}")
                
                st.subheader("📋 Sample Data")
                st.dataframe(data.head(10))
            
            with tab2:
                # Feature correlation heatmap
                feature_cols = ['packet_size', 'connection_duration', 'bytes_per_second', 
                              'unique_destinations', 'port_scan_attempts', 'failed_connections',
                              'dns_queries', 'protocol_anomalies']
                
                corr_matrix = data[feature_cols].corr()
                fig = px.imshow(corr_matrix, 
                               title="Feature Correlation Matrix",
                               color_continuous_scale="RdBu",
                               text_auto=True)
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Traffic patterns visualization
                col1_viz, col2_viz = st.columns(2)
                
                with col1_viz:
                    fig1 = px.box(data, x='label', y='bytes_per_second', 
                                 title="Bytes per Second by Traffic Type")
                    fig1.update_xaxis(tickvals=[0, 1], ticktext=['Normal', 'Botnet'])
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2_viz:
                    fig2 = px.box(data, x='label', y='unique_destinations', 
                                 title="Unique Destinations by Traffic Type")
                    fig2.update_xaxis(tickvals=[0, 1], ticktext=['Normal', 'Botnet'])
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Distribution plots
                fig3 = px.histogram(data, x='port_scan_attempts', color='label', 
                                   title="Port Scan Attempts Distribution",
                                   barmode='overlay', opacity=0.7)
                st.plotly_chart(fig3, use_container_width=True)
        
        # Model Performance
        if st.session_state.models_trained:
            st.subheader("🎯 Model Performance")
            
            # Evaluate models
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # ML model predictions
            ml_predictions = st.session_state.robot.ml_model.predict(X_test)
            ml_proba = st.session_state.robot.ml_model.predict_proba(X_test)[:, 1]
            
            # "DL" model predictions
            dl_predictions = st.session_state.robot.dl_model.predict(X_test)
            dl_proba = st.session_state.robot.dl_model.predict_proba(X_test)[:, 1]
            
            # Display metrics
            col1_metrics, col2_metrics = st.columns(2)
            
            with col1_metrics:
                st.write("**🌳 Random Forest Performance:**")
                ml_accuracy = accuracy_score(y_test, ml_predictions)
                ml_report = classification_report(y_test, ml_predictions, output_dict=True)
                
                st.metric("Accuracy", f"{ml_accuracy:.3f}")
                st.metric("Precision", f"{ml_report['1']['precision']:.3f}")
                st.metric("Recall", f"{ml_report['1']['recall']:.3f}")
                st.metric("F1-Score", f"{ml_report['1']['f1-score']:.3f}")
            
            with col2_metrics:
                st.write("**🧮 Logistic Regression Performance:**")
                dl_accuracy = accuracy_score(y_test, dl_predictions)
                dl_report = classification_report(y_test, dl_predictions, output_dict=True)
                
                st.metric("Accuracy", f"{dl_accuracy:.3f}")
                st.metric("Precision", f"{dl_report['1']['precision']:.3f}")
                st.metric("Recall", f"{dl_report['1']['recall']:.3f}")
                st.metric("F1-Score", f"{dl_report['1']['f1-score']:.3f}")
            
            # Feature importance
            st.subheader("🔍 Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': st.session_state.feature_cols,
                'importance': st.session_state.robot.ml_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance, x='importance', y='feature', 
                        orientation='h', title="Random Forest Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, ml_predictions)
            fig = px.imshow(cm, text_auto=True, 
                           title="Confusion Matrix - Random Forest",
                           labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Real-time Detection Panel
        st.subheader("🔍 Real-time Detection")
        
        if st.session_state.models_trained:
            # Detection controls
            st.write("**🎮 Detection Controls**")
            
            # Manual detection
            if st.button("🔍 Scan Single Sample", type="primary"):
                sample_data = st.session_state.data.sample(1)
                feature_cols = st.session_state.feature_cols
                
                sample_features = st.session_state.robot.scaler.transform(sample_data[feature_cols])
                result = st.session_state.robot.simulate_real_time_detection(sample_features)
                
                # Display results
                if result['is_botnet']:
                    st.error("⚠️ **BOTNET DETECTED!**")
                    st.balloons()
                else:
                    st.success("✅ **Normal Traffic**")
                
                # Show prediction scores
                st.write("**🎯 Prediction Scores:**")
                st.progress(result['ml_prediction'], f"Random Forest: {result['ml_prediction']:.3f}")
                st.progress(result['dl_prediction'], f"Logistic Reg: {result['dl_prediction']:.3f}")
                st.progress(result['ensemble_prediction'], f"Ensemble: {result['ensemble_prediction']:.3f}")
                
                # Show sample features
                st.write("**📊 Sample Features:**")
                for i, col in enumerate(feature_cols):
                    st.write(f"• {col}: {sample_data[col].iloc[0]:.2f}")
            
            # Auto detection toggle
            st.write("**🔄 Auto Detection**")
            if 'auto_detect' not in st.session_state:
                st.session_state.auto_detect = False
            
            if st.button("▶️ Start Auto Detection" if not st.session_state.auto_detect else "⏸️ Stop Auto Detection"):
                st.session_state.auto_detect = not st.session_state.auto_detect
            
            # Auto detection display
            if st.session_state.auto_detect:
                detection_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                # Initialize counters
                if 'detection_count' not in st.session_state:
                    st.session_state.detection_count = 0
                    st.session_state.botnet_detected = 0
                
                # Simulate detection
                sample_data = st.session_state.data.sample(1)
                feature_cols = st.session_state.feature_cols
                sample_features = st.session_state.robot.scaler.transform(sample_data[feature_cols])
                result = st.session_state.robot.simulate_real_time_detection(sample_features)
                
                st.session_state.detection_count += 1
                if result['is_botnet']:
                    st.session_state.botnet_detected += 1
                
                # Update display
                with detection_placeholder.container():
                    if result['is_botnet']:
                        st.error("⚠️ BOTNET DETECTED!")
                    else:
                        st.success("✅ Normal Traffic")
                    
                    st.progress(result['ensemble_prediction'], f"Threat Level: {result['ensemble_prediction']:.3f}")
                
                with metrics_placeholder.container():
                    st.write("**📈 Detection Statistics:**")
                    st.metric("Total Scans", st.session_state.detection_count)
                    st.metric("Threats Found", st.session_state.botnet_detected)
                    detection_rate = st.session_state.botnet_detected / st.session_state.detection_count
                    st.metric("Detection Rate", f"{detection_rate:.2%}")
                
                # Auto refresh
                time.sleep(detection_speed)
                st.rerun()
            
            # Reset counters
            if st.button("🔄 Reset Statistics"):
                st.session_state.detection_count = 0
                st.session_state.botnet_detected = 0
                st.success("Statistics reset!")
        
        else:
            st.info("🔧 **Setup Required**\n\n1. Generate network data\n2. Train models\n3. Start detection")
            
            # Show progress
            progress_val = 0
            if st.session_state.data_generated:
                progress_val = 0.5
            if st.session_state.models_trained:
                progress_val = 1.0
            
            st.progress(progress_val, f"Setup Progress: {progress_val*100:.0f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🔧 Robot Simulation Features")
    
    col1_feat, col2_feat, col3_feat = st.columns(3)
    
    with col1_feat:
        st.markdown("""
        **🧠 AI Models:**
        - Random Forest Classifier
        - Logistic Regression
        - Isolation Forest (Anomaly Detection)
        """)
    
    with col2_feat:
        st.markdown("""
        **📊 Detection Features:**
        - Real-time monitoring
        - Ensemble predictions
        - Feature importance analysis
        """)
    
    with col3_feat:
        st.markdown("""
        **🎯 Capabilities:**
        - Synthetic data generation
        - Model performance metrics
        - Interactive visualization
        """)

if __name__ == "__main__":
    main()
