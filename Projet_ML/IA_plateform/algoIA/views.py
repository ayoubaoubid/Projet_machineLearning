import os
import joblib
import numpy as np
import pandas as pd

from django.shortcuts import render
from django.conf import settings
from .forms import UploadCSVForm
from .cleaning import Data_Cleaning

# ==========================================
#              GENERAL & UTILS
# ==========================================

def index(request):
    return render(request, 'index.html')

def load_models(name):
    # os.path.abspath() donne le chemin de fichier actuel (views.py -> __file__)
    # os.path.dirname() revient en arrière (cd ..). 
    # Exemple: base_dir = C:/Users/Hiba/project/app/views.py -> C:/Users/Hiba/project/app -> C:/Users/Hiba/project/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # os.path.join() fait la concaténation.
    # Resultat: C:/Users/Hiba/project/models_ai
    models_dir = os.path.join(base_dir, 'models_ai')

    # model_path va avoir C:/Users/Hiba/project/models_ai/Model_ent.pkl
    model_path = os.path.join(models_dir, name)

    # joblib.load() fait l'importation
    ml_model = joblib.load(model_path) 
    return ml_model


# ==========================================
#      RÉGRESSION LOGISTIQUE (Classif)
# ==========================================

def regLog_details(request):
    return render(request, 'regLog_details.html')

def regLog_atelier(request):
    return render(request, 'regLog_atelier.html')

def regLog_form(request):
    return render(request, 'regLog_form.html')

def regLog_prediction(request):
    if request.method == 'POST':
        pregnancies = float(request.POST.get('Pregnancies'))
        glucose = float(request.POST.get('Glucose'))
        blood_pressure = float(request.POST.get('Blood_Pressure'))
        skin_thickness = float(request.POST.get('Skin_Thickness'))
        insulin = float(request.POST.get('Insulin'))
        bmi = float(request.POST.get('Bmi'))
        diabetes_pedigree = float(request.POST.get('Diabetes_Pedigree'))
        Age = int(request.POST.get('Age'))

        model = load_models('CM.pkl')
        theta_opti = model['theta']
        sc = model['ss']

        def segmoind(z):
            return 1 / (1 + np.exp(-z))
        
        def model_func(x, theta):
            fc = x.dot(theta)
            return segmoind(fc)
        
        new_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, Age, 1]).reshape(1, -1)
        new_data_scaled = sc.transform(new_data)
 
        prediction = model_func(new_data_scaled, theta_opti)
        
        predicted_class = 1 if prediction[0] >= 0.8 else 0

        context = {
            'prediction': predicted_class,
            'input_data': {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'Blood_Pressure': blood_pressure,
                'Skin_Thickness': skin_thickness,
                'Insulin': insulin,
                'Bmi': bmi,
                'Diabetes_Pedigree': diabetes_pedigree,
                'Age': Age
            }
        }
        return render(request, 'classification_result.html', context)
        
    return render(request, 'regLog_form.html')


# ==========================================
#           RANDOM FOREST
# ==========================================

def randomFor_details(request):
    return render(request, 'randomFor_details.html')

# --- Random Forest Classification ---

def randomFor_cla_atelier(request):
    return render(request, 'randomFor_cla_atelier.html')

def randomFor_cla_form(request):
    return render(request, 'randomFor_cla_form.html')

def randomFor_cla_prediction(request):
    if request.method == 'POST':
        pregnancies = float(request.POST.get('Pregnancies'))
        glucose = float(request.POST.get('Glucose'))
        blood_pressure = float(request.POST.get('Blood_Pressure'))
        skin_thickness = float(request.POST.get('Skin_Thickness'))
        insulin = float(request.POST.get('Insulin'))
        bmi = float(request.POST.get('Bmi'))
        diabetes_pedigree = float(request.POST.get('Diabetes_Pedigree'))
        Age = int(request.POST.get('Age'))
        
        model = load_models('randomForest_class.pkl')

        prediction_RF = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, Age]])
        prediction_value = int(prediction_RF[0])  # 0 ou 1

        context = {
            'prediction': prediction_value,
            'input_data': {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'Blood_Pressure': blood_pressure,
                'Skin_Thickness': skin_thickness,
                'Insulin': insulin,
                'Bmi': bmi,
                'Diabetes_Pedigree': diabetes_pedigree,
                'Age': Age
            }
        }
        return render(request, 'classification_result.html', context)
        
    return render(request, 'randomFor_cla_form.html')

# --- Random Forest Regression ---

def randomForest_reg_atelier(request):
    return render(request, 'randomForest_reg_atelier.html')

def randomForest_reg_form(request):
    return render(request, 'randomForest_reg_form.html')

def randomFor_reg_prediction(request):
    if request.method == 'POST':
        Trip_Distance_km = float(request.POST.get('Trip_Distance_km'))
        Time_of_Day = str(request.POST.get('Time_of_Day'))
        Day_of_Week = str(request.POST.get('Day_of_Week'))
        Passenger_Count = float(request.POST.get('Passenger_Count'))
        Traffic_Conditions = str(request.POST.get('Traffic_Conditions'))
        Weather = str(request.POST.get('Weather'))
        Base_Fare = float(request.POST.get('Base_Fare'))
        Per_Km_Rate = float(request.POST.get('Per_Km_Rate'))
        Per_Minute_Rate = float(request.POST.get('Per_Minute_Rate'))
        Trip_Duration_Minutes = int(request.POST.get('Trip_Duration_Minutes'))
        
        type_Time_of_Day = {'Morning': 2, 'Afternoon': 0, 'Evening': 1, 'Night': 3}
        type_Day_of_Week = {'weekday': 0, 'weekend': 1}
        type_Traffic_Conditions = {'low': 1, 'high': 0, 'medium': 2}
        type_Weather = {'clear': 0, 'rain': 1, 'snow': 2}
        
        model = load_models('randomForest_reg.pkl')

        prediction_RF = model.predict([[
            Trip_Distance_km, Passenger_Count, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes, 
            type_Time_of_Day[Time_of_Day], type_Day_of_Week[Day_of_Week], 
            type_Traffic_Conditions[Traffic_Conditions], type_Weather[Weather]
        ]])
        
        predicted_class = prediction_RF[0]

        # Mapping inverse pour affichage
        time_mapping = {"Morning": 'Matin', "Afternoon": 'Après-midi', "Evening": 'Soir', "Night": 'Nuit'}
        day_mapping = {"weekday": 'Jour de semaine', "weekend": 'Week-end'}
        traffic_mapping = {"high": 'Élevée', "low": 'Faible', "medium": 'Moyenne'}
        weather_mapping = {"clear": 'Clair', "rain": 'Pluie', "snow": 'Neige'}

        context = {
            'predicted_price': round(predicted_class, 2),
            'input_data': {
                'Trip_Distance_km': Trip_Distance_km,
                'Time_of_Day': time_mapping[Time_of_Day],
                'Day_of_Week': day_mapping[Day_of_Week],
                'Passenger_Count': Passenger_Count, 
                'Traffic_Conditions': traffic_mapping[Traffic_Conditions],
                'Weather': weather_mapping[Weather],
                'Base_Fare': Base_Fare,
                'Per_Km_Rate': Per_Km_Rate,
                'Per_Minute_Rate': Per_Minute_Rate,
                'Trip_Duration_Minutes': Trip_Duration_Minutes,
            },
        }
        return render(request, 'regression_result.html', context)
    
    return render(request, 'randomForest_reg_form.html')


# ==========================================
#                XG-BOOST
# ==========================================

def XGboost_details(request):
    return render(request, 'XGboost_details.html' )

def XGboost_reg_atelier(request):
    return render(request, 'XGboost_reg_atelier.html' )

def XGboost_reg_form(request):
    return render(request, 'XGboost_reg_form.html')

def XGboost_cla_atelier(request):
    return render(request, 'XGboost_cla_atelier.html')

def XGboost_cla_form(request):
    return render(request, 'XGboost_cla_form.html')

# --- XGBoost Regression ---

def XGboost_reg_prediction(request):
    if request.method == 'POST':
        Trip_Distance_km = float(request.POST.get('Trip_Distance_km'))
        Time_of_Day = str(request.POST.get('Time_of_Day'))
        Day_of_Week = str(request.POST.get('Day_of_Week'))
        Passenger_Count = float(request.POST.get('Passenger_Count'))
        Traffic_Conditions = str(request.POST.get('Traffic_Conditions'))
        Weather = str(request.POST.get('Weather'))
        Base_Fare = float(request.POST.get('Base_Fare'))
        Per_Km_Rate = float(request.POST.get('Per_Km_Rate'))
        Per_Minute_Rate = float(request.POST.get('Per_Minute_Rate'))
        Trip_Duration_Minutes = int(request.POST.get('Trip_Duration_Minutes'))
        
        type_Time_of_Day = {'Morning': 2, 'Afternoon': 0, 'Evening': 1, 'Night': 3}
        type_Day_of_Week = {'weekday': 0, 'weekend': 1}
        type_Traffic_Conditions = {'low': 1, 'high': 0, 'medium': 2}
        type_Weather = {'clear': 0, 'rain': 1, 'snow': 2}
        
        model = load_models('XGboost_R.pkl')

        prediction_RF = model.predict([[
            Trip_Distance_km, Passenger_Count, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes, 
            type_Time_of_Day[Time_of_Day], type_Day_of_Week[Day_of_Week], 
            type_Traffic_Conditions[Traffic_Conditions], type_Weather[Weather]
        ]])
        
        predicted_class = prediction_RF[0]

        time_mapping = {"Morning": 'Matin', "Afternoon": 'Après-midi', "Evening": 'Soir', "Night": 'Nuit'}
        day_mapping = {"weekday": 'Jour de semaine', "weekend": 'Week-end'}
        traffic_mapping = {"high": 'Élevée', "low": 'Faible', "medium": 'Moyenne'}
        weather_mapping = {"clear": 'Clair', "rain": 'Pluie', "snow": 'Neige'}

        context = {
            'predicted_price': round(predicted_class, 2),
            'input_data': {
                'Trip_Distance_km': Trip_Distance_km,
                'Time_of_Day': time_mapping[Time_of_Day],
                'Day_of_Week': day_mapping[Day_of_Week],
                'Passenger_Count': Passenger_Count, 
                'Traffic_Conditions': traffic_mapping[Traffic_Conditions],
                'Weather': weather_mapping[Weather],
                'Base_Fare': Base_Fare,
                'Per_Km_Rate': Per_Km_Rate,
                'Per_Minute_Rate': Per_Minute_Rate,
                'Trip_Duration_Minutes': Trip_Duration_Minutes,
            },
        }
        return render(request, 'regression_result.html', context)
    
    return render(request, 'XGboost_reg_form.html')

# --- XGBoost Classification ---

def XGboost_cla_prediction(request):
    if request.method == 'POST':
        pregnancies = float(request.POST.get('Pregnancies'))
        glucose = float(request.POST.get('Glucose'))
        blood_pressure = float(request.POST.get('Blood_Pressure'))
        skin_thickness = float(request.POST.get('Skin_Thickness'))
        insulin = float(request.POST.get('Insulin'))
        bmi = float(request.POST.get('Bmi'))
        diabetes_pedigree = float(request.POST.get('Diabetes_Pedigree'))
        Age = int(request.POST.get('Age'))
        
        model = load_models('XGboost_C.pkl')

        prediction_RF = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, Age]])
        prediction_value = int(prediction_RF[0])  # 0 ou 1

        context = {
            'prediction': prediction_value,
            'input_data': {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'Blood_Pressure': blood_pressure,
                'Skin_Thickness': skin_thickness,
                'Insulin': insulin,
                'Bmi': bmi,
                'Diabetes_Pedigree': diabetes_pedigree,
                'Age': Age
            }
        }
        return render(request, 'classification_result.html', context)
        
    return render(request, 'XGboost_cla_form.html')


# ==========================================
#           RÉGRESSION LINÉAIRE
# ==========================================

def reg_lin_details(request):
    return render(request, 'reg_lin_details.html')

def reg_lin_atelier(request):
    return render(request, 'reg_lin_atelier.html')

def reg_lin_form(request):
    return render(request, 'reg_lin_form.html')

def reg_lin_pred(request):
    if request.method == 'POST':
        model_used = request.POST.get("model")

        trip_distance = float(request.POST.get('Trip_Distance_km'))
        time_of_day = request.POST.get('Time_of_Day')
        day_of_week = request.POST.get('Day_of_Week')
        passenger_count = int(request.POST.get('Passenger_Count'))
        traffic_conditions = request.POST.get('Traffic_Conditions')
        weather = request.POST.get('Weather')
        base_fare = float(request.POST.get('Base_Fare'))
        per_km_rate = float(request.POST.get('Per_Km_Rate'))
        per_minute_rate = float(request.POST.get('Per_Minute_Rate'))
        trip_duration = float(request.POST.get('Trip_Duration_Minutes'))

        time_mapping_num = {"Morning": 2, "Afternoon": 0, "Evening": 1, "Night": 3}
        day_mapping_num = {"weekday": 0, "weekend": 1}
        traffic_mapping_num = {"high": 0, "low": 1, "medium": 2}
        weather_mapping_num = {"clear": 0, "rain": 1, "snow": 2}

        time_num = time_mapping_num[time_of_day]
        day_num = day_mapping_num[day_of_week]
        traffic_num = traffic_mapping_num[traffic_conditions]
        weather_num = weather_mapping_num[weather]

        features = [[
            trip_distance, time_num, day_num, passenger_count, traffic_num, weather_num,
            base_fare, per_km_rate, per_minute_rate, trip_duration
        ]]
        
        # Model 1
        model1 = load_models("RLM.pkl")
        model1.scaling(features)
        prediction1 = model1.predict(np.hstack((model1.x_scaled, np.ones((1,1)))))
        predicted_price1 = prediction1[0][0]
 
        # Model 2
        model2 = load_models("RLSL.pkl")
        prediction2 = model2.predict(model1.x_scaled)
        predicted_price2 = prediction2[0]

        # Mapping UI
        time_mapping_ui = {"Morning": 'Matin', "Afternoon": 'Après-midi', "Evening": 'Soir', "Night": 'Nuit'}
        day_mapping_ui = {"weekday": 'Jour de semaine', "weekend": 'Week-end'}
        traffic_mapping_ui = {"high": 'Élevée', "low": 'Faible', "medium": 'Moyenne'}
        weather_mapping_ui = {"clear": 'Clair', "rain": 'Pluie', "snow": 'Neige'}

        context = {
            'predicted_price': [round(predicted_price1, 6), round(predicted_price2, 6)],
            'input_data': {
                'Trip_Distance_km': trip_distance,
                'Time_of_Day': time_mapping_ui[time_of_day],
                'Day_of_Week': day_mapping_ui[day_of_week],
                'Passenger_Count': passenger_count,
                'Traffic_Conditions': traffic_mapping_ui[traffic_conditions],
                'Weather': weather_mapping_ui[weather],
                'Base_Fare': base_fare,
                'Per_Km_Rate': per_km_rate,
                'Per_Minute_Rate': per_minute_rate,
                'Trip_Duration_Minutes': trip_duration
            }
        }
        return render(request, 'regression_lin_result.html', context)

    return render(request, 'reg_lin_form.html')


# ==========================================
#                   SVM
# ==========================================

def SVM_details(request):
    return render(request, 'SVM_details.html')

# --- SVR (Regression) ---

def SVR_atelier(request):
    return render(request, 'SVR_atelier.html')

def SVR_form(request):
    return render(request, 'SVR_form.html')

def SVR_pred(request):
    if request.method == 'POST':
        model_used = request.POST.get("model")

        trip_distance = float(request.POST.get('Trip_Distance_km'))
        time_of_day = request.POST.get('Time_of_Day')
        day_of_week = request.POST.get('Day_of_Week')
        passenger_count = int(request.POST.get('Passenger_Count'))
        traffic_conditions = request.POST.get('Traffic_Conditions')
        weather = request.POST.get('Weather')
        base_fare = float(request.POST.get('Base_Fare'))
        per_km_rate = float(request.POST.get('Per_Km_Rate'))
        per_minute_rate = float(request.POST.get('Per_Minute_Rate'))
        trip_duration = float(request.POST.get('Trip_Duration_Minutes'))

        time_mapping_num = {"Morning": 2, "Afternoon": 0, "Evening": 1, "Night": 3}
        day_mapping_num = {"weekday": 0, "weekend": 1}
        traffic_mapping_num = {"high": 0, "low": 1, "medium": 2}
        weather_mapping_num = {"clear": 0, "rain": 1, "snow": 2}

        time_num = time_mapping_num[time_of_day]
        day_num = day_mapping_num[day_of_week]
        traffic_num = traffic_mapping_num[traffic_conditions]
        weather_num = weather_mapping_num[weather]

        features = [[
            trip_distance, time_num, day_num, passenger_count, traffic_num, weather_num,
            base_fare, per_km_rate, per_minute_rate, trip_duration
        ]]
        
        model = load_models("SVR.pkl")
        model_temp = load_models("RLM.pkl") # Used for scaling
        model_temp.scaling(features)
 
        prediction = model.predict(model_temp.x_scaled)
        predicted_price = prediction[0]

        time_mapping_ui = {"Morning": 'Matin', "Afternoon": 'Après-midi', "Evening": 'Soir', "Night": 'Nuit'}
        day_mapping_ui = {"weekday": 'Jour de semaine', "weekend": 'Week-end'}
        traffic_mapping_ui = {"high": 'Élevée', "low": 'Faible', "medium": 'Moyenne'}
        weather_mapping_ui = {"clear": 'Clair', "rain": 'Pluie', "snow": 'Neige'}

        context = {
            'predicted_price': round(predicted_price, 10),
            'input_data': {
                'Trip_Distance_km': trip_distance,
                'Time_of_Day': time_mapping_ui[time_of_day],
                'Day_of_Week': day_mapping_ui[day_of_week],
                'Passenger_Count': passenger_count,
                'Traffic_Conditions': traffic_mapping_ui[traffic_conditions],
                'Weather': weather_mapping_ui[weather],
                'Base_Fare': base_fare,
                'Per_Km_Rate': per_km_rate,
                'Per_Minute_Rate': per_minute_rate,
                'Trip_Duration_Minutes': trip_duration
            }
        }
        return render(request, 'regression_result.html', context)

    return render(request, 'SVR_form.html')


# --- SVC (Classification) ---

def SVC_atelier(request):
    return render(request, 'SVC_atelier.html')

def SVC_form(request):
    return render(request, 'SVC_form.html')

def SVC_pred(request):
    if request.method == 'POST':
        pregnancies = float(request.POST.get('Pregnancies'))
        glucose = float(request.POST.get('Glucose'))
        blood_pressure = float(request.POST.get('Blood_Pressure'))
        skin_thickness = float(request.POST.get('Skin_Thickness'))
        insulin = float(request.POST.get('Insulin'))
        bmi = float(request.POST.get('Bmi'))
        diabetes_pedigree = float(request.POST.get('Diabetes_Pedigree'))
        age = float(request.POST.get('Age'))

        features = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]

        model = load_models("SVC.pkl")
        prediction = model.predict(features)
        prediction_value = int(prediction[0])  # 0 ou 1

        context = {
            'prediction': prediction_value,
            'input_data': {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'Blood_Pressure': blood_pressure,
                'Skin_Thickness': skin_thickness,
                'Insulin': insulin,
                'Bmi': bmi,
                'Diabetes_Pedigree': diabetes_pedigree,
                'Age': age
            }
        }
        return render(request, 'classification_result.html', context)

    return render(request, 'SVC_form.html')


# ==========================================
#          DECISION TREES (DT)
# ==========================================

def DT_details(request):
    return render(request, 'DT_details.html' )

def DT_reg_atelier(request):
    return render(request, 'DT_reg_atelier.html' )

def DT_reg_form(request):
    return render(request, 'DT_reg_form.html')

def DT_cla_atelier(request):
    return render(request, 'DT_cla_atelier.html')

def DT_cla_form(request):
    return render(request, 'DT_cla_form.html')

# --- DT Regression ---

def DT_reg_prediction(request):
    if request.method == 'POST':
        Trip_Distance_km = float(request.POST.get('Trip_Distance_km'))
        Time_of_Day = str(request.POST.get('Time_of_Day'))
        Day_of_Week = str(request.POST.get('Day_of_Week'))
        Passenger_Count = float(request.POST.get('Passenger_Count'))
        Traffic_Conditions = str(request.POST.get('Traffic_Conditions'))
        Weather = str(request.POST.get('Weather'))
        Base_Fare = float(request.POST.get('Base_Fare'))
        Per_Km_Rate = float(request.POST.get('Per_Km_Rate'))
        Per_Minute_Rate = float(request.POST.get('Per_Minute_Rate'))
        Trip_Duration_Minutes = int(request.POST.get('Trip_Duration_Minutes'))
        
        type_Time_of_Day = {'Morning': 2, 'Afternoon': 0, 'Evening': 1, 'Night': 3}
        type_Day_of_Week = {'weekday': 0, 'weekend': 1}
        type_Traffic_Conditions = {'low': 1, 'high': 0, 'medium': 2}
        type_Weather = {'clear': 0, 'rain': 1, 'snow': 2}
        
        model = load_models('DTR.pkl')

        prediction_RF = model.predict([[
            Trip_Distance_km, Passenger_Count, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes, 
            type_Time_of_Day[Time_of_Day], type_Day_of_Week[Day_of_Week], 
            type_Traffic_Conditions[Traffic_Conditions], type_Weather[Weather]
        ]])
        
        predicted_class = prediction_RF[0]

        time_mapping_ui = {"Morning": 'Matin', "Afternoon": 'Après-midi', "Evening": 'Soir', "Night": 'Nuit'}
        day_mapping_ui = {"weekday": 'Jour de semaine', "weekend": 'Week-end'}
        traffic_mapping_ui = {"high": 'Élevée', "low": 'Faible', "medium": 'Moyenne'}
        weather_mapping_ui = {"clear": 'Clair', "rain": 'Pluie', "snow": 'Neige'}

        context = {
            'predicted_price': round(predicted_class, 2),
            'input_data': {
                'Trip_Distance_km': Trip_Distance_km,
                'Time_of_Day': time_mapping_ui[Time_of_Day],
                'Day_of_Week': day_mapping_ui[Day_of_Week],
                'Passenger_Count': Passenger_Count, 
                'Traffic_Conditions': traffic_mapping_ui[Traffic_Conditions],
                'Weather': weather_mapping_ui[Weather],
                'Base_Fare': Base_Fare,
                'Per_Km_Rate': Per_Km_Rate,
                'Per_Minute_Rate': Per_Minute_Rate,
                'Trip_Duration_Minutes': Trip_Duration_Minutes,
            },
        }
        return render(request, 'regression_result.html', context)
    
    return render(request, 'DT_reg_form.html')

# --- DT Classification ---

def DT_cla_prediction(request):
    if request.method == 'POST':
        pregnancies = float(request.POST.get('Pregnancies'))
        glucose = float(request.POST.get('Glucose'))
        blood_pressure = float(request.POST.get('Blood_Pressure'))
        skin_thickness = float(request.POST.get('Skin_Thickness'))
        insulin = float(request.POST.get('Insulin'))
        bmi = float(request.POST.get('Bmi'))
        diabetes_pedigree = float(request.POST.get('Diabetes_Pedigree'))
        Age = int(request.POST.get('Age'))
        
        model = load_models('DTC.pkl')

        prediction_RF = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, Age]])
        prediction_value = int(prediction_RF[0])  # 0 ou 1

        context = {
            'prediction': prediction_value,
            'input_data': {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'Blood_Pressure': blood_pressure,
                'Skin_Thickness': skin_thickness,
                'Insulin': insulin,
                'Bmi': bmi,
                'Diabetes_Pedigree': diabetes_pedigree,
                'Age': Age
            }
        }
        return render(request, 'classification_result.html', context)
        
    return render(request, 'DT_cla_form.html')


# ==========================================
#          INFORMATIONS & DONNÉES
# ==========================================

def informations(request):
    return render(request, 'informations.html')

def add_info_form(request):
    return render(request, 'add_info_form.html')

def add_info_done(request):
    return render(request, 'add_info_done.html')

def add_info(request):
    if request.method == "POST":
        data = {
            "Trip_Distance_km": [request.POST.get("Trip_Distance_km")],
            "Time_of_Day": [request.POST.get("Time_of_Day")],
            "Day_of_Week": [request.POST.get("Day_of_Week")],
            "Passenger_Count": [request.POST.get("Passenger_Count")],
            "Traffic_Conditions": [request.POST.get("Traffic_Conditions")],
            "Weather": [request.POST.get("Weather")],
            "Base_Fare": [request.POST.get("Base_Fare")],
            "Per_Km_Rate": [request.POST.get("Per_Km_Rate")],
            "Per_Minute_Rate": [request.POST.get("Per_Minute_Rate")],
            "Trip_Duration_Minutes": [request.POST.get("Trip_Duration_Minutes")],
            "Trip_Price": [request.POST.get("Trip_Total_Cost")]
        }
        
        folder_path = os.path.join(settings.MEDIA_ROOT, 'new_data')
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "taxi_new_data.xlsx")
        
        df_new = pd.DataFrame(data)
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df = pd.concat([df, df_new], axis=0)
        else:
            df = df_new
        
        df.to_excel(file_path, index=False, header=True)
        
        return render(request, "add_info_done.html")
    
    return render(request, "add_info_form.html")



# ==========================================
#           NETTOYAGE (CLEANING)
# ==========================================

def cleaning_form(request):
    return render(request, 'cleaning_form.html')

def cleaning_atelier(request):
    return render(request, 'cleaning_atelier.html')

def cleaning_done(request):
    return render(request, 'cleaning_done.html')

def cleaning_proc(request):
    if request.method == "POST":
        form = UploadCSVForm(request.POST, request.FILES)

        if form.is_valid():
            csv_file = request.FILES['csv_file']

            reply = request.POST.get('has_header')
            sep = request.POST.get('delimiter')
            
            # Initialisation
            clean = Data_Cleaning(csv_file, sep=sep)

            # Configuration
            clean.reply = reply
            clean.sep = sep
            cols = request.POST.get('column_names')
            clean.cols = cols
            
            # Étapes de nettoyage
            clean.isHeader()
            clean.separation_xnum_xstr()
            clean.extraction_date()
            clean.encodage()

            clean.target = request.POST.get('id_name')
            clean.suppression_id()

            reply = request.POST.get('imputation_method')
            clean.reply = reply
            clean.val_manq()

            clean.duplication()
            clean.remp_outlier()

            clean.target = request.POST.get('target_column')
            clean.separation_x_y()

            reply = request.POST.get('standardize')
            clean.reply = reply
            clean.standarisation()

            # Sauvegarde
            df_final = clean.df_final()
            file_path = os.path.join(settings.MEDIA_ROOT, 'cleaned', 'dataset_cleaned.csv')
            df_final.to_csv(file_path, index=False)

            file_url = os.path.join(settings.MEDIA_URL, 'cleaned', 'dataset_cleaned.csv')
            print("✅ Fichier prêt à être sauvegardé ici :", file_path)

            context = {'download_url': file_url}
            return render(request, "cleaning_done.html", context)

    return render(request,"cleaning_form.html")