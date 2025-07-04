from flask import Flask, request, render_template
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html')  # Entry page

@app.route('/home')
def home():
    return render_template('home.html')  # Prediction form page

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')

    # POST: retrieve form data
    data = CustomData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("race_ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        writing_score=int(request.form.get("writing_score")),
        reading_score=int(request.form.get("reading_score"))
    )
    pred_df = data.get_data_as_data_frame()
    results = PredictPipeline().predict(pred_df)[0]

    return render_template('home.html', results=results)

@app.route('/healthz')
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

