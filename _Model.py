import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

#---------------------------------------------
# Define states
#---------------------------------------------

OFF, IDLE, ACTIVE = 0, 1, 2
np.random.seed(42)

# Time range (5 hours of 1-min readings)
timestamp = pd.date_range('2025-01-01', periods=3000, freq='1min')

# Simulate realistic state transitions (machines don’t jump wildly)
states = []
current_state = np.random.choice([OFF, IDLE, ACTIVE])
for _ in range(len(timestamp)):
    # 5% chance to change state at each minute
    if np.random.rand() < 0.05:
        current_state = np.random.choice([OFF, IDLE, ACTIVE],p=[0.2, 0.3, 0.5])
    states.append(current_state)
states = np.array(states)

#---------------------------------------------
# Generate features with overlap and noise
#---------------------------------------------

rms = np.zeros(len(states))
volt = np.zeros(len(states))
speed = np.zeros(len(states))

for i, s in enumerate(states):
    if s == OFF:
        rms[i] = np.random.normal(0.2, 0.1)
        volt[i] = np.random.normal(12.3, 0.15)
        speed[i] = 0 + np.random.normal(0, 0.1) 

    elif s == IDLE:
        rms[i] = np.random.normal(0.7, 0.25)
        volt[i] = np.random.normal(13.6, 0.1)
        speed[i] = np.random.normal(3, 0.5)  

    else:  # ACTIVE
        rms[i] = np.random.normal(1.3, 0.35)
        volt[i] = np.random.normal(13.8, 0.1)
        speed[i] = abs(np.random.normal(12, 5)) 

# Add random short voltage dips (simulate battery drain under load)
for _ in range(3):
    start = np.random.randint(100, len(volt) - 200)
    duration = np.random.randint(30, 80)
    volt[start:start+duration] -= np.linspace(0, 0.8, duration)

# Clip voltage to realistic range
volt = np.clip(volt, 11.5, 14.0)


#---------------------------------------------
# Build DataFrame
#---------------------------------------------

df = pd.DataFrame({
    'Timestamp': timestamp,
    'RMS': rms,
    'volt': volt,
    'speed': speed,
    'State': states
})


#---------------------------------------------
# Data Preprocessing
#---------------------------------------------

df['Timestamp']=df['Timestamp'].astype(np.int64)//10**9
x=df.iloc[:,1:-1]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#---------------------------------------------
# Random Forest Classifier
#---------------------------------------------

rf=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=5,min_samples_split=5,min_samples_leaf=5)
rf.fit(x_train,y_train)
Acc_rf=rf.score(x_train,y_train)
Acc_rf_test=rf.score(x_test,y_test)

# print("Random Forest Classifier Score is:",Acc_rf*100,
#       "Random Forest Classifier Test Score is:",Acc_rf_test*100)



#---------------------------------------------
# UNit test
#---------------------------------------------

def unit_test(model):
    pre_off=model.predict([[0.1, 12.2, 0.1]])[0]
    assert pre_off==0, f"expected 0 but got {pre_off}"

    pre_idle=model.predict([[0.7, 13.8, 2.5]])[0]
    assert pre_idle==1, f"expected 1 but got {pre_idle}"

    pre_active=model.predict([[1.5, 13.9, 9]])[0]
    assert pre_active==2, f"expected 2 but got {pre_active}"

    print("All Unit Tests are Passed Successfully")

unit_test(rf)


#---------------------------------------------
# Save trained Random Forest model to a file
#---------------------------------------------

joblib.dump(rf, "sensors_model.pkl")
