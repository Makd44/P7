import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import shap

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

df = pd.read_csv('df_final.csv', sep=',',nrows=100 )
df= df.drop('TARGET', axis =1)

@app.route('/prediction', methods=['POST'])
def prediction():
    
    data = request.get_json(force=True)
    prediction = model.predict_proba(df[df['SK_ID_CURR']== data['SK_ID_CURR']])
    P = np.array(prediction).flatten()
    output = P[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)