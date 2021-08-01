# Import libraries
from types import MethodDescriptorType
from flask import Flask,render_template,request
import joblib

# create instance of an app
app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')
@app.route('/')
def welcome():
    return render_template('input.html')

@app.route('/input',methods=['POST'])
def inputs():
    a = request.form.get('preg')
    b = request.form.get('plas')
    c = request.form.get('pres')
    e = request.form.get('test')
    f = request.form.get('mass')
    g = request.form.get('pedi')
    h = request.form.get('age')
    pred = model.predict([[int(a),int(b),int(c),int(e),int(f),int(g),int(h)]])
    if(pred[0]):
        output = 'diabetic'
    else:
        output='non-diabetic'
    return render_template('output.html',predicted_text = f'Person is {output}')



# Run app
if __name__=='__main__':
    app.run(debug=True)