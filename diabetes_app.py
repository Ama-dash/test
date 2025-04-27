import tkinter as tk
import numpy as np
import joblib

# Load your saved model and scaler
classifier = joblib.load('trained_model.sav')
scalar = joblib.load('scaler.sav')

def predict_diabetes():
    try:
        # Get values from user inputs
        input_data = [float(entry.get()) for entry in entries]
        input_data_np = np.asarray(input_data).reshape(1, -1)

        # Standardize and predict
        std_data = scalar.transform(input_data_np)
        prediction = classifier.predict(std_data)

        # Show result
        if prediction[0] == 0:
            result_label.config(text="The person is NOT diabetic.")
        else:
            result_label.config(text="The person IS diabetic.")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

# Create the GUI window
root = tk.Tk()
root.title("Diabetes Prediction App")

labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
entries = []

for label in labels:
    tk.Label(root, text=label).pack()
    entry = tk.Entry(root)
    entry.pack()
    entries.append(entry)

predict_button = tk.Button(root, text="Predict", command=predict_diabetes)
predict_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
