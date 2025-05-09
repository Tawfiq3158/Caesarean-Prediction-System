# Caesarean Section Classification

This project uses machine learning to predict the likelihood of a Caesarean section (C-section) during childbirth based on maternal health data. The model is built using logistic regression and deployed via a Flask web application to provide real-time predictions and data-driven insights for medical professionals.

## Project Features:
- **Predict C-section likelihood** based on key maternal health indicators (age, delivery history, blood pressure, heart condition, etc.).
- **Flask Web App**: Allows users to input data and receive a prediction.
- **Real-time Visualizations**: Displays prediction results, data overview, and statistical summaries.
- **Data Preprocessing**: Includes feature encoding and data normalization for better model performance.

## Technologies Used:
- Python
- Flask
- scikit-learn (for the logistic regression model)
- Pandas (for data manipulation)
- Matplotlib/Seaborn (for visualizations)
- HTML, CSS, and JavaScript (for web interface)

## Setup and Installation:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tawfiq3158/caesarean-section-classification.git
2. Navigate to the project folder:
   cd caesarean-section-classification
3. Install the required dependencies:
   pip install -r requirements.txt
4. Run the Flask app:
   python app.py
5. Access the application in your browser at http://127.0.0.1:5000/.
