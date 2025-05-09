from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('caesarean.pkl', 'rb'))

# Helper function to load tables
def load_tables():
    try:
        df_head = pd.read_csv('static/df_head.csv')
        df_stats = pd.read_csv('static/df_stats.csv')
    except:
        df_head, df_stats = None, None
    return df_head, df_stats

@app.route('/')
def index():
    df_head, df_stats = load_tables()
    return render_template("index.html",
                           prediction=None,
                           head_table=df_head.to_html(classes='table table-bordered', index=False) if df_head is not None else "",
                           stats_table=df_stats.to_html(classes='table table-bordered', index=False) if df_stats is not None else "",
                           request=request)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        age = int(request.form['age'])
        delivery_no = int(request.form['delivery_no'])
        delivery = request.form['delivery']
        blood_pressure = request.form['bp']
        heart_problem = request.form['heart']

        # Manual encoding
        bp_low = 1 if blood_pressure.lower() == "low" else 0
        bp_normal = 1 if blood_pressure.lower() == "normal" else 0
        heart_inept = 1 if heart_problem.lower() == "inept" else 0
        delivery_premature = 1 if delivery.lower() == "premature" else 0
        delivery_timely = 1 if delivery.lower() == "timely" else 0

        # Build input
        input_data = pd.DataFrame([[age, delivery_no, delivery_premature, delivery_timely, bp_low, bp_normal, heart_inept]],
                                  columns=['Age', 'Delivey No', 'Delivery No_Premature', 'Delivery No_Timely',
                                           'Blood of Pressure_Low', 'Blood of Pressure_Normal', 'Heart Problem_inept'])

        prediction = model.predict(input_data)[0]
        prediction_result = "Yes - High likelihood of Caesarean section." if prediction == 1 else "No - Low likelihood of Caesarean section."

        # Reload tables
        df_head, df_stats = load_tables()

        return render_template('index.html',
                               prediction=prediction_result,
                               head_table=df_head.to_html(classes='table table-bordered', index=False) if df_head is not None else "",
                               stats_table=df_stats.to_html(classes='table table-bordered', index=False) if df_stats is not None else "",
                               request=request)

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction="Error: Something went wrong. Please try again.", request=request)

if __name__ == '__main__':
    app.run(debug=True)
