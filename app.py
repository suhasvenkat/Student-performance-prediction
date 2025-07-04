from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Extract form inputs
            gender = request.form.get('gender')
            race_ethnicity = request.form.get('race_ethnicity')
            parental_level_of_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test_preparation_course')
            writing_score = request.form.get('writing_score')
            reading_score = request.form.get('reading_score')

            # Check for missing inputs
            if not all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, writing_score, reading_score]):
                return render_template("home.html", results="Please fill all the fields!")

            # Dummy prediction logic (replace with your model)
            predicted_score = (int(reading_score) + int(writing_score)) // 2

            return render_template("home.html", results=predicted_score)
        
        except Exception as e:
            print("Exception in /predict:", e)
            return render_template("home.html", results=f"Error: {str(e)}")

    # Important: Provide results=None in GET mode to avoid template issues
    return render_template("home.html", results=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
