import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load base dataset and trained model once
df_1 = pd.read_csv("first_telc.csv")
model = pickle.load(open("modelBuild.sav", "rb"))

@app.route("/")
def loadPage():
    return render_template("home.html")


@app.route("/", methods=["POST"])
def predict():

    # Fetch form values
    try:
        inputQuery1 = int(request.form['query1'])  # SeniorCitizen
        inputQuery2 = float(request.form['query2'])  # MonthlyCharges
        inputQuery3 = float(request.form['query3'])  # TotalCharges
        inputQuery4 = request.form['query4']
        inputQuery5 = request.form['query5']
        inputQuery6 = request.form['query6']
        inputQuery7 = request.form['query7']
        inputQuery8 = request.form['query8']
        inputQuery9 = request.form['query9']
        inputQuery10 = request.form['query10']
        inputQuery11 = request.form['query11']
        inputQuery12 = request.form['query12']
        inputQuery13 = request.form['query13']
        inputQuery14 = request.form['query14']
        inputQuery15 = request.form['query15']
        inputQuery16 = request.form['query16']
        inputQuery17 = request.form['query17']
        inputQuery18 = request.form['query18']
        inputQuery19 = int(request.form['query19'])  # tenure

    except Exception as e:
        return render_template("home.html", output1="Error reading input.", output2=str(e))

    # Prepare new input row
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7,
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                         'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                         'DeviceProtection', 'TechSupport', 'StreamingTV', 
                                         'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    # Merge with original DF for consistent encoding
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Create tenure groups
    labels = ["{0} - {1}".format(i, i+11) for i in range(1, 72, 12)]
    df_2["tenure_group"] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=["tenure"], inplace=True)

    # One-hot encoding
    df_encoded = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                      'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                      'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

    # Align columns to match model
    df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    single_prediction = model.predict(df_encoded.tail(1))
    probability = model.predict_proba(df_encoded.tail(1))[:, 1][0]

    if single_prediction == 1:
        o1 = "⚠ Customer will likely churn."
    else:
        o1 = "✔ Customer will likely stay."

    o2 = f"Confidence: {round(probability*100, 2)} %"

    return render_template("home.html", output1=o1, output2=o2)


if __name__ == "__main__":
    app.run(debug=True)
