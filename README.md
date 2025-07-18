# 🤖 Botnet Detection Robot Simulation

An AI-powered network security monitoring system that uses machine learning and deep learning to detect botnet activities in real-time.

## Features

- **Hybrid Detection System**: Combines Random Forest, Neural Networks, and Isolation Forest
- **Real-time Monitoring**: Live botnet detection simulation
- **Interactive Dashboard**: Streamlit-based web interface
- **Synthetic Data Generation**: Realistic network traffic simulation
- **Performance Analytics**: Model evaluation and training metrics
- **Anomaly Detection**: Unsupervised learning for unknown threats

## Architecture

### Machine Learning Components
- **Random Forest**: Ensemble classifier for botnet detection
- **Isolation Forest**: Anomaly detection for unknown threats
- **Neural Network**: Deep learning model with dropout regularization
- **Ensemble Prediction**: Combined scoring from multiple models

### Robot Simulation Features
- Real-time network traffic analysis
- Automated threat detection
- Performance monitoring
- Interactive visualization dashboard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/botnet-detection-robot.git
cd botnet-detection-robot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Generate Data**: Click "Generate Network Data" to create synthetic network traffic
2. **Train Models**: Click "Train Models" to train ML and DL models
3. **Real-time Detection**: Enable "Auto Detection" or use "Scan Single Sample"

## Model Performance

The system achieves high accuracy through:
- Feature engineering on network traffic patterns
- Ensemble learning combining multiple algorithms
- Real-time anomaly detection
- Continuous model evaluation

## Deployment

### GitHub Pages
The app can be deployed using Streamlit Cloud:
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from GitHub

### Docker Deployment
```bash
# Build Docker image
docker build -t botnet-detection-robot .

# Run container
docker run -p 8501:8501 botnet-detection-robot
```

## Technical Details

### Data Features
- Packet size analysis
- Connection duration patterns
- Bytes per second metrics
- Unique destination counting
- Port scan detection
- Failed connection monitoring
- DNS query analysis
- Protocol anomaly detection

### ML/DL Models
- **Random Forest**: 100 estimators, handles non-linear patterns
- **Neural Network**: 4 hidden layers with dropout (0.3, 0.3, 0.2)
- **Isolation Forest**: Contamination rate 0.1 for anomaly detection
- **Ensemble**: Weighted average of ML and DL predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Streamlit for the web framework
- TensorFlow for deep learning capabilities
- Scikit-learn for machine learning algorithms
- Plotly for interactive visualizations
