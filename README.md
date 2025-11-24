# 🧠 Mental Health Status Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**AI-powered mental health assessment based on lifestyle & technology usage patterns**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model-details)

</div>

---

## 📖 Overview

An intelligent machine learning application that predicts mental health status (Good/Moderate/Poor) by analyzing lifestyle factors and technology usage patterns. Built with **Random Forest Classifier** and deployed using **Streamlit** for an interactive, user-friendly experience.

### 🎯 Key Highlights

- 🤖 **High Accuracy Prediction** - Trained on 10,000+ real-world data samples
- 🎨 **Beautiful UI** - Modern, gradient-styled interface with interactive visualizations
- 🔒 **100% Private** - All processing happens locally, no data leaves your device
- 📊 **Feature Importance** - Understand which factors most influence your mental health
- ⚡ **Real-time Results** - Instant predictions with confidence scores
- 💡 **Personalized Recommendations** - Get actionable advice based on your assessment

---

## ✨ Features

### 🖥️ Interactive Dashboard
- **Sliders & Controls** - Easy input for 12 lifestyle variables
- **Live Predictions** - Instant mental health status classification
- **Probability Charts** - Visual confidence distribution across categories
- **Gradient Cards** - Beautiful info cards with smooth animations

### 📈 Data Analysis
- **Feature Importance Visualization** - Horizontal bar charts showing key factors
- **Performance Metrics** - Model accuracy, precision, and recall
- **Comprehensive Reporting** - Classification reports and confusion matrices

### 🎯 Smart Insights
- **Stress Level Detection** - Automatic stress status evaluation
- **Anxiety Assessment** - Heuristic-based anxiety level estimation
- **Custom Recommendations** - Tailored advice based on predicted status

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Ayuuu-tech/Mental-Health-Status-Prediction.git
cd Mental-Health-Status-Prediction
```

2. **Create virtual environment**
```bash
python -m venv .venv
```

3. **Activate virtual environment**

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Train the model (Run in Jupyter)**
```bash
jupyter notebook "Mini Project.ipynb"
# Execute cell 78: Mental Health Prediction Model
```

6. **Launch Streamlit app**
```bash
streamlit run app.py
```

7. **Open in browser**
```
http://localhost:8501
```

---

## 📊 Model Details

### Algorithm
**Random Forest Classifier** - An ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

### Features (12 Variables)
| Category | Features |
|----------|----------|
| 👤 **Personal** | Age, Gender |
| ⏱️ **Tech Usage** | Technology Hours, Social Media Hours, Gaming Hours, Screen Time |
| 😴 **Wellness** | Sleep Hours, Physical Activity Hours |
| 😰 **Stress** | Stress Level |
| 🤝 **Support** | Support Systems Access, Work Environment Impact, Online Support Usage |

### Performance
- **Training Data**: 10,000+ samples
- **Train/Test Split**: 80/20 with stratification
- **Classes**: 3 (Good, Moderate, Poor)
- **Accuracy**: Check notebook for detailed metrics

### Model Architecture
```
Input (12 features) 
    ↓
Label Encoding (Categorical → Numeric)
    ↓
Random Forest (200 trees, max_depth=15)
    ↓
Output (3 classes with probabilities)
```

---

## 🎨 Screenshots

### Main Dashboard
Beautiful gradient header with AI-powered mental health assessment interface.

### Prediction Results
- ✅ **Good** - Green success message with confetti animation
- ⚠️ **Moderate** - Yellow warning with monitoring advice
- 🔴 **Poor** - Red alert with professional support recommendations

### Model Insights
- Interactive feature importance charts
- Performance metrics cards
- Gradient info boxes explaining privacy and methodology

---

## 💻 Usage

### Step 1: Input Your Information
Use the sidebar sliders to enter your data:
- Personal details (Age, Gender)
- Daily technology usage hours
- Sleep and exercise patterns
- Stress levels and support systems

### Step 2: Get Prediction
Click the **"🔮 Predict Mental Health Status"** button to receive:
- Primary mental health classification
- Confidence probability distribution
- Stress and anxiety indicators
- Personalized recommendations

### Step 3: Explore Insights
Scroll down to view:
- Feature importance rankings
- Model performance metrics
- Research-based key insights
- Privacy information

---

## 📁 Project Structure

```
Mental-Health-Status-Prediction/
├── 📓 Mini Project.ipynb          # Complete analysis & model training
├── 🐍 app.py                      # Streamlit web application
├── 📊 cleaned_data.csv            # Preprocessed dataset
├── 📋 requirements.txt            # Python dependencies
├── 📁 models/                     # Saved model pickles
│   └── mh_model_*.pkl
├── 📁 plots/                      # Generated visualizations
│   ├── correlation_heatmap.html
│   └── interactive_dashboard.html
└── 📖 README.md                   # This file
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) | Core programming language |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data manipulation & analysis |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) | Numerical computations |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) | Machine learning algorithms |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Web app framework |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) | Interactive notebooks |

---

## 📈 Future Enhancements

- [ ] 🌐 Multi-language support (Hindi, Spanish, etc.)
- [ ] 📱 Mobile-responsive design optimization
- [ ] 🔄 Real-time model retraining with user feedback
- [ ] 📊 Advanced visualizations with Plotly
- [ ] 🧪 A/B testing different ML algorithms
- [ ] 💾 Export reports as PDF
- [ ] 🔔 Mental health tracking over time
- [ ] 🤝 Integration with mental health resources

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

---

## ⚠️ Disclaimer

**Important Notice:**

This application is designed for **informational and educational purposes only**. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.

- ❌ **Not a diagnostic tool** - Results are predictions based on ML models
- 🏥 **Seek professional help** - Always consult qualified mental health professionals
- 🔒 **Privacy matters** - While data is processed locally, use responsibly
- 📊 **Statistical predictions** - Model accuracy may vary for individuals

If you or someone you know is experiencing a mental health crisis:
- 🇺🇸 **USA**: National Suicide Prevention Lifeline - 988
- 🇮🇳 **India**: AASRA - 91-22-27546669
- 🌍 **International**: [findahelpline.com](https://findahelpline.com)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Ayush**

- 🐙 GitHub: [@Ayuuu-tech](https://github.com/Ayuuu-tech)
- 📧 Email: [Your Email]
- 💼 LinkedIn: [Your LinkedIn]

---

## 🌟 Acknowledgments

- Dataset contributors and mental health research community
- Streamlit team for the amazing framework
- scikit-learn developers for ML tools
- Open source community for inspiration

---

<div align="center">

### ⭐ Star this repo if you find it helpful!

**Made with ❤️ and 🧠 by Ayush**

[⬆ Back to Top](#-mental-health-status-predictor)

</div>
