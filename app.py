import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, text

#################################################
# Database Setup
#################################################
engine = create_engine("sqlite:///database/loan_application_data.db")

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

# get the project path
basedir = os.path.abspath(os.path.dirname(__file__))

### sqlalchemy config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir,
                                                                    'loan_application_data.db') + '?check_same_thread=False'
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# show raw sql when set true
app.config['SQLALCHEMY_ECHO'] = True
### define type
db = SQLAlchemy(app)
# init database session
Session = db.sessionmaker(bind=db.engine)
session = Session()


#################################################
# Flask Routes
#################################################
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/index.html")
def returnhome():
    return render_template("index.html")


@app.route("/form.html")
def form():
    return render_template("form.html")


@app.route("/form.html", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        array = [x for x in request.form.values()]
        print(input)
        input_df = pd.DataFrame({'business_or_commercial': [array[0]],
                                 'loan_amount': [array[1]],
                                 'rate_of_interest': [array[2]],
                                 'term': [array[3]],
                                 'interest_only': [array[4]],
                                 'property_value': [array[5]],
                                 'income': [array[6]],
                                 'credit_score': [array[7]],
                                 'age': [array[8]],
                                 'ltv': [array[9]],
                                 'dtir1': [array[10]]})
        X_scaler = joblib.load('model/standardized.pkl')
        model = joblib.load('model/XGBoost.sav')
        print(len(input_df.columns))
        scaled_values = X_scaler.transform(np.array(input_df))
        predict = model.predict(scaled_values)
        if predict[0] == 1:
            result = "Your Loan is approved"
        else:
            result = "Your Loan is not approved"
    return render_template("form.html", result=result)


@app.route("/pairplot.html")
def compare():
    return render_template("pairplot.html")


# go to index.html web page
@app.route('/matrix.html')
def index():  # put application's code here
    return render_template("matrix.html")


# data model
class Loan_Data(db.Model):
    __tablename__ = "loan_data"
    index = db.Column(db.Integer, primary_key=True)
    ID = db.Column(db.TEXT)
    business_or_commercial = db.Column(db.TEXT)
    loan_amount = db.Column(db.TEXT)
    rate_of_interest = db.Column(db.TEXT)
    term = db.Column(db.TEXT)
    interest_only = db.Column(db.TEXT)
    property_value = db.Column(db.TEXT)
    income = db.Column(db.TEXT)
    credit_score = db.Column(db.TEXT)
    age = db.Column(db.TEXT)
    ltv = db.Column(db.TEXT)
    dtir1 = db.Column(db.TEXT)
    status = db.Column(db.TEXT)


class Matrix_Data(db.Model):
    __tablename__ = "mytrix_data"
    index = db.Column(db.Integer, primary_key=True)
    BaslineName = db.Column(db.TEXT)
    Accuracy = db.Column(db.REAL)
    Precision = db.Column(db.REAL)
    MCC = db.Column(db.REAL)
    PPV = db.Column(db.REAL)
    NPV = db.Column(db.REAL)
    Recall = db.Column(db.REAL)
    F1 = db.Column(db.REAL)
    color = db.Column(db.TEXT)


# go to dashboard.html web page
@app.route('/dashboard.html')
def dashboard():  # put application's code here
    return render_template("dashboard.html")


@app.route('/matrix')
def matrix():
    data_list = getAllMytrixData()
    return_data = []
    for obj in data_list:
        obj_data = {
            "BaslineName": obj.BaslineName,
            "Accuracy": obj.Accuracy,
            "Precision": obj.Precision,
            "MCC": obj.MCC,
            "PPV": obj.PPV,
            "NPV": obj.NPV,
            "Recall": obj.Recall,
            "F1": obj.F1,
        }
        return_data.append(obj_data)

    return json.dumps({
        'data': return_data,
        'msg': 'success',
        'code': '0000'
    })


# return charts data
@app.route('/data/charts', methods=['GET'])
def dataCharts():  # put application's code here
    bar_yAxis = ['<25', '25-34', '35-44', '45-54', '55-64', '65-74', '>74']
    bar_series = []

    # 5 : <25
    # 0 : 25-34
    # 1 : 35-44
    # 2 : 45-54
    # 3 : 55-64
    # 4 : 65-74
    # 6 : >74
    # define age group array
    age_groups = [5, 0, 1, 2, 3, 4, 6]
    # find each rate of each age
    for age in age_groups:
        value = selectPassRate(age)
        print('age=[' + str(age) + '] pass rate=' + str(value))
        bar_series.append(value)

    # define bar data structure
    barData = {
        'yAxis': bar_yAxis,
        'series': bar_series,
    }

    # define pie data structure
    pieData = [
        # CREDIT SCORE
        [
            {'name': 'Below 600', 'value': selectCreditScore(None, 600)},
            {'name': '600-800', 'value': selectCreditScore(600, 800)},
            {'name': 'Above 800', 'value': selectCreditScore(800, None)},
        ],
        # LTV RATIO
        [
            {'name': 'Above 90%', 'value': selectLTVRATIO(90, None)},
            {'name': '80%-90%', 'value': selectLTVRATIO(80, 90)},
            {'name': '60%-80%', 'value': selectLTVRATIO(60, 80)},
            {'name': '40%-60%', 'value': selectLTVRATIO(40, 60)},
            {'name': 'Below 40%', 'value': selectLTVRATIO(None, 40)},
        ],
        # DTI RATIO
        [
            {'name': 'Above 45%', 'value': selectDTIRATIO(45, None)},
            {'name': '45%-35%', 'value': selectDTIRATIO(35, 45)},
            {'name': 'Below 35%', 'value': selectDTIRATIO(None, 35)},
        ]
    ]

    return_data = {
        'barData': barData,
        'pieData': pieData,
    }
    return json.dumps({
        'data': return_data,
        'msg': 'success',
        'code': '0000'
    })


### sqlite database operations
# find all data of mytrix data
def getAllMytrixData():
    try:
        obj_list = Matrix_Data.query.all()
        return obj_list
    except Exception as e:
        res = {'code': 0, 'message': 'find data list error'}
        return json.dumps(res, ensure_ascii=False, indent=4)


# find pass rate and its count value
def selectPassRate(age):
    age = str(age)
    try:
        sql = 'select (select count(*) from loan_data where age = ' + age + ' and status = 0) * 1.0 ' \
                                                                            '/ (select count(*) from loan_data where age = ' + age + ')'
        results = session.execute(text(sql)).first()
        return round(results[0] * 100, 2)
    except Exception as e:
        res = {'code': 0, 'message': 'find data list error'}
        return json.dumps(res, ensure_ascii=False, indent=4)


# find all data of specific credit score and its count value 
def selectCreditScore(min, max):
    try:
        sub_min_sql = ''
        sub_max_sql = ''
        if min is not None:
            sub_min_sql = ' and credit_score > ' + str(min)
        if max is not None:
            sub_max_sql = ' and credit_score < ' + str(max)

        sql = 'select (select count(*) from loan_data ' \
              'where 1=1 ' + sub_min_sql + sub_max_sql + ' ) * 1.0 / (select count(*) from loan_data)'
        results = session.execute(text(sql)).first()
        return round(results[0] * 100, 2)
    except Exception as e:
        res = {'code': 0, 'message': 'find data list error'}
        return json.dumps(res, ensure_ascii=False, indent=4)


# find all data of specific ltv and its count value
def selectLTVRATIO(min, max):
    try:
        sub_min_sql = ''
        sub_max_sql = ''
        if min is not None:
            sub_min_sql = ' and ltv > ' + str(min)
        if max is not None:
            sub_max_sql = ' and ltv < ' + str(max)

        sql = 'select (select count(*) from loan_data ' \
              'where 1=1 ' + sub_min_sql + sub_max_sql + ' ) * 1.0 / (select count(*) from loan_data)'
        results = session.execute(text(sql)).first()
        return round(results[0] * 100, 2)
    except Exception as e:
        res = {'code': 0, 'message': 'find data list error'}
        return json.dumps(res, ensure_ascii=False, indent=4)


# find all data of specific dtir and dtir count value
def selectDTIRATIO(min, max):
    try:
        sub_min_sql = ''
        sub_max_sql = ''
        if min is not None:
            sub_min_sql = ' and dtir1 > ' + str(min)
        if max is not None:
            sub_max_sql = ' and dtir1 < ' + str(max)

        sql = 'select (select count(*) from loan_data ' \
              'where 1=1 ' + sub_min_sql + sub_max_sql + ' ) * 1.0 / (select count(*) from loan_data)'
        results = session.execute(text(sql)).first()
        return round(results[0] * 100, 2)
    except Exception as e:
        res = {'code': 0, 'message': 'find data list error'}
        return json.dumps(res, ensure_ascii=False, indent=4)


# data for chartx
# @app.route("/api/chartx")
# def chartx():
#     database_df = pd.read_sql(f"Select loan_amount, status from loan_data",engine)
#     database_JSON = database_df.to_json(orient = 'records')
#     return database_JSON

if __name__ == "__main__":
    app.run()
