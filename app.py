import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configure the page
st.set_page_config(page_title="Botnet Detection", layout="wide")

# Title
st.title("🤖 Botnet Detection in Robotic Network Logs")

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("synthetic_robot_logs.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file 'synthetic_robot_logs.csv' not found. Please upload the file.")
        return None

# Load model function (with error handling)
@st.cache_resource
def load_trained_model():
    try:
        from tensorflow.keras.models import load_model
        model = load_model("botnet_detection_model.h5")
        return model
    except Exception as e:
        st.warning(f"⚠️ Pre-trained model not found: {e}")
        return None

# Create a simple mock model for demonstration
def create_mock_model():
    """Creates a simple mock prediction function when the trained model is not available"""
    def mock_predict(data):
        # Simple heuristic: larger packets with smaller intervals are more likely to be botnets
        packet_size = data[0][0]
        interval = data[0][1]
        
        # Normalize inputs roughly
        normalized_score = (packet_size / 1000) * (1 / (interval + 0.1))
        probability = min(max(normalized_score / 10, 0.1), 0.9)
        
        return [[probability]]
    
    class MockModel:
        def predict(self, data):
            return mock_predict(data)
    
    return MockModel()

# Load data
df = load_data()

if df is not None:
    # Display basic info about the dataset
    st.subheader("📊 Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        if 'is_botnet' in df.columns:
            botnet_count = df['is_botnet'].sum()
            st.metric("Botnet Records", botnet_count)
    
    with col3:
        if 'is_botnet' in df.columns:
            normal_count = len(df) - df['is_botnet'].sum()
            st.metric("Normal Records", normal_count)
    
    # Dataset preview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())
    
    # Check if required columns exist
    required_columns = ['packet_size', 'interval']
    if all(col in df.columns for col in required_columns):
        # Initialize scaler
        scaler = StandardScaler()
        X = df[required_columns].values
        X_scaled = scaler.fit_transform(X)
        
        # Load or create model
        model = load_trained_model()
        if model is None:
            st.info("ℹ️ Using mock model for demonstration (train the model first for accurate predictions)")
            model = create_mock_model()
        
        # User input for prediction
        st.subheader("🔍 Predict Botnet Activity")
        
        col1, col2 = st.columns(2)
        with col1:
            packet_size = st.number_input(
                "Packet Size", 
                min_value=0.0, 
                max_value=float(df['packet_size'].max()) if 'packet_size' in df.columns else 10000.0,
                value=float(df['packet_size'].mean()) if 'packet_size' in df.columns else 500.0,
                step=1.0
            )
        
        with col2:
            interval = st.number_input(
                "Interval", 
                min_value=0.0, 
                max_value=float(df['interval'].max()) if 'interval' in df.columns else 100.0,
                value=float(df['interval'].mean()) if 'interval' in df.columns else 5.0,
                step=0.1
            )
        
        if st.button("🔮 Predict", type="primary"):
            try:
                # Prepare input data
                input_data = np.array([[packet_size, interval]])
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0][0]
                
                # Display result
                if prediction > 0.5:
                    st.error(f"🔴 **Botnet Activity Detected!** (Confidence: {prediction:.2%})")
                else:
                    st.success(f"🟢 **Normal Activity** (Confidence: {(1-prediction):.2%})")
                
                # Show prediction details
                st.info(f"Raw prediction score: {prediction:.4f}")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
        
        # Visualization section
        st.subheader("📊 Data Visualization")
        
        if 'is_botnet' in df.columns and len(df) > 0:
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Distributions", "Class Balance"])
            
            with tab1:
                st.write("**Packet Size vs Interval Scatter Plot**")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create scatter plot
                scatter = sns.scatterplot(
                    data=df, 
                    x="packet_size", 
                    y="interval", 
                    hue="is_botnet", 
                    palette={0: "green", 1: "red"},
                    alpha=0.6,
                    ax=ax
                )
                
                # Customize the plot
                ax.set_xlabel("Packet Size")
                ax.set_ylabel("Interval")
                ax.set_title("Packet Size vs Interval (by Botnet Status)")
                
                # Update legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, ['Normal', 'Botnet'], title='Activity Type')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                st.write("**Feature Distributions**")
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Packet size distribution
                for class_val in [0, 1]:
                    subset = df[df['is_botnet'] == class_val]
                    axes[0].hist(subset['packet_size'], alpha=0.7, 
                               label=f"{'Normal' if class_val == 0 else 'Botnet'}", bins=30)
                axes[0].set_xlabel("Packet Size")
                axes[0].set_ylabel("Frequency")
                axes[0].set_title("Packet Size Distribution")
                axes[0].legend()
                
                # Interval distribution
                for class_val in [0, 1]:
                    subset = df[df['is_botnet'] == class_val]
                    axes[1].hist(subset['interval'], alpha=0.7, 
                               label=f"{'Normal' if class_val == 0 else 'Botnet'}", bins=30)
                axes[1].set_xlabel("Interval")
                axes[1].set_ylabel("Frequency")
                axes[1].set_title("Interval Distribution")
                axes[1].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                st.write("**Class Distribution**")
                class_counts = df['is_botnet'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['lightgreen', 'lightcoral']
                    ax.pie(class_counts.values, labels=['Normal', 'Botnet'], 
                          autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title("Class Distribution")
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(6, 6))
                    bars = ax.bar(['Normal', 'Botnet'], class_counts.values, 
                                 color=['lightgreen', 'lightcoral'])
                    ax.set_ylabel("Count")
                    ax.set_title("Class Counts")
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    plt.close()
        
        else:
            st.info("ℹ️ Visualization requires 'is_botnet' column in the dataset")
    
    else:
        st.error(f"❌ Required columns missing. Expected: {required_columns}, Found: {list(df.columns)}")

else:
    st.info("👆 Please ensure 'synthetic_robot_logs.csv' is in the same directory as this app.")
    
    # Show sample data format
    st.subheader("📋 Expected Data Format")
    sample_data = pd.DataFrame({
        'packet_size': [512, 1024, 256, 2048],
        'interval': [2.5, 1.2, 5.8, 0.8],
        'is_botnet': [0, 1, 0, 1]
    })
    st.dataframe(sample_data)
    st.write("- `packet_size`: Size of the network packet")
    st.write("- `interval`: Time interval between packets")  
    st.write("- `is_botnet`: Target variable (0=Normal, 1=Botnet)")

# Sidebar with information
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app demonstrates botnet detection using machine learning on robotic network logs.

**Features:**
- Dataset exploration
- Real-time prediction
- Interactive visualizations
- Model performance metrics

**Usage:**
1. Ensure data file is present
2. Train model using the provided script
3. Enter packet size and interval values
4. Get botnet detection prediction
""")

st.sidebar.header("📁 Required Files")
st.sidebar.write("""
- `synthetic_robot_logs.csv` - Dataset
- `botnet_detection_model.h5` - Trained model
""")
