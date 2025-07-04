from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import os

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

##route for home page
@app.route('/')
def root():
    return render_template('index.html')  # or: return redirect('/home')


@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        form_data = request.form.to_dict()
        print("[DEBUG] Received form data:", form_data)

        # ✅ Check for missing or empty fields
        missing_fields = [key for key, value in form_data.items() if not value]
        if missing_fields:
            error_msg = f"Error: Missing values for {', '.join(missing_fields)}"
            return render_template('home.html', results=error_msg)

        try:
            # ✅ Cast scores to float (or int) to ensure proper typing
            data = CustomData(
                gender=form_data.get("gender"),
                race_ethnicity=form_data.get("race_ethnicity"),
                parental_level_of_education=form_data.get("parental_level_of_education"),
                lunch=form_data.get("lunch"),
                test_preparation_course=form_data.get("test_preparation_course"),
                writing_score=float(form_data.get("writing_score")),
                reading_score=float(form_data.get("reading_score"))
            )

            pred_df = data.get_data_as_data_frame()
            print("[DEBUG] DataFrame passed to model:\n", pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0])

        except Exception as e:
            return render_template('home.html', results=f"Prediction failed: {str(e)}")

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


