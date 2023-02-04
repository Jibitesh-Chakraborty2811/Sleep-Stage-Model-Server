from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from tensorflow import keras
import joblib
import mysql.connector

mydb = mysql.connector.connect(
host="localhost",
user="root",
password="Jibimax123",
)

mycursor = mydb.cursor()
mycursor.execute("USE UserPassWordDB")



app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')


@app.route("/login")
def login():
    return render_template('t_login.html')



@app.route("/upload",methods=['GET','POST'])
def techlogin():
    if request.method == 'POST':
        mycursor.execute("select * from TechnicianLogin")
        #mydb.commit()
        user = request.form.get('username')
        password = request.form.get('password')
        flag = 0
        for x in mycursor:
            if x[0] == user and x[1] == password:
                flag = flag+1

        if flag != 0:
            return render_template('upload.html')
    return ""

@app.route("/success",methods=['GET','POST'])
def new():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        raw = mne.io.read_raw_edf(f.filename)
        data = raw.get_data()
        df = pd.DataFrame(data)
        df = df.transpose()
        df_stats = df.describe()
        df_stats = df_stats.transpose()
        def norm(x):
            return (x-df_stats['mean'])/df_stats['std']
        df = norm(df)
        X = df.iloc[25770:55770]
        ann = keras.models.load_model('Sleep8.h5')
        dt = joblib.load('SleepDT.joblib')
        knn = joblib.load('SleepKNN.joblib')
        nb = joblib.load('SleepNB.joblib')
        rf = joblib.load('SleepRF.joblib')
        svm = joblib.load('SleepSVM_RBF.joblib')

        Y_ANN = np.round(ann.predict(X))
        Y_DT = dt.predict(X)
        Y_KNN = knn.predict(X)
        Y_NB = nb.predict(X)
        Y_RF = rf.predict(X)
        Y_SVM = svm.predict(X)

        count0_ANN = 0
        count1_ANN = 0
        count2_ANN = 0
        count3_ANN = 0
        count4_ANN = 0
        ANN_OUT = []
        
        for i in range(0,30000):
            if(Y_ANN[i][0] == 1):
                count0_ANN = count0_ANN+1
                ANN_OUT.append(0)
            elif(Y_ANN[i][1] == 1):
                count1_ANN = count1_ANN+1
                ANN_OUT.append(1)
            elif(Y_ANN[i][2] == 1):
                count2_ANN = count2_ANN+1
                ANN_OUT.append(2)
            elif(Y_ANN[i][3] == 1):
                count3_ANN = count3_ANN+1
                ANN_OUT.append(3)
            elif(Y_ANN[i][4] == 1):
                count4_ANN = count4_ANN+1
                ANN_OUT.append(4)
            else:
                ANN_OUT.append(-1)
        
        count0_DT = 0
        count1_DT = 0
        count2_DT = 0
        count3_DT = 0
        count4_DT = 0

        for i in range(0,30000):
            if(Y_DT[i] == 0):
                count0_DT = count0_DT+1
            elif(Y_DT[i] == 1):
                count1_DT = count1_DT+1
            elif(Y_DT[i] == 2):
                count2_DT = count2_DT+1
            elif(Y_DT[i] == 3):
                count3_DT = count3_DT+1
            elif(Y_DT[i] == 4):
                count4_DT = count4_DT+1
        
        count0_KNN = 0
        count1_KNN = 0
        count2_KNN = 0
        count3_KNN = 0
        count4_KNN = 0

        for i in range(0,30000):
            if(Y_KNN[i] == 0):
                count0_KNN = count0_KNN+1
            elif(Y_KNN[i] == 1):
                count1_KNN = count1_KNN+1
            elif(Y_KNN[i] == 2):
                count2_KNN = count2_KNN+1
            elif(Y_KNN[i] == 3):
                count3_KNN = count3_KNN+1
            elif(Y_KNN[i] == 4):
                count4_KNN = count4_KNN+1
        
        count0_NB = 0
        count1_NB = 0
        count2_NB = 0
        count3_NB = 0
        count4_NB = 0

        for i in range(0,30000):
            if(Y_NB[i] == 0):
                count0_NB = count0_NB+1
            elif(Y_NB[i] == 1):
                count1_NB = count1_NB+1
            elif(Y_NB[i] == 2):
                count2_NB = count2_NB+1
            elif(Y_NB[i] == 3):
                count3_NB = count3_NB+1
            elif(Y_NB[i] == 4):
                count4_NB = count4_NB+1

        count0_RF = 0
        count1_RF = 0
        count2_RF = 0
        count3_RF = 0
        count4_RF = 0

        for i in range(0,30000):
            if(Y_RF[i] == 0):
                count0_RF = count0_RF+1
            elif(Y_RF[i] == 1):
                count1_RF = count1_RF+1
            elif(Y_RF[i] == 2):
                count2_RF = count2_RF+1
            elif(Y_RF[i] == 3):
                count3_RF = count3_RF+1
            elif(Y_RF[i] == 4):
                count4_RF = count4_RF+1
        
        count0_SVM = 0
        count1_SVM = 0
        count2_SVM = 0
        count3_SVM = 0
        count4_SVM = 0

        for i in range(0,30000):
            if(Y_SVM[i] == 0):
                count0_SVM = count0_SVM+1
            elif(Y_SVM[i] == 1):
                count1_SVM = count1_SVM+1
            elif(Y_SVM[i] == 2):
                count2_SVM = count2_SVM+1
            elif(Y_SVM[i] == 3):
                count3_SVM = count3_SVM+1
            elif(Y_SVM[i] == 4):
                count4_SVM = count4_SVM+1
        
        #count0 = (count0_ANN + count0_DT + count0_KNN + count0_NB + count0_RF + count0_SVM)/6
        #count1 = (count1_ANN + count1_DT + count1_KNN + count1_NB + count1_RF + count1_SVM)/6
        #count2 = (count2_ANN + count2_DT + count2_KNN + count2_NB + count2_RF + count2_SVM)/6
        #count3 = (count3_ANN + count3_DT + count3_KNN + count3_NB + count3_RF + count3_SVM)/6
        #count4 = (count4_ANN + count4_DT + count4_KNN + count4_NB + count4_RF + count4_SVM)/6     

        def majority(a,b,c,d,e,f):
            count0 = 0
            count1 = 1
            count2 = 2
            count3 = 3
            count4 = 4

            if(a == 0):
                count0 = count0 + 1
            if(a == 1):
                count1 = count1 + 1
            if(a == 2):
                count2 = count2 + 1
            if(a == 3):
                count3 = count3 + 1
            if(a == 4):
                count4 = count4 + 1

            if(b == 0):
                count0 = count0 + 1
            if(b == 1):
                count1 = count1 + 1
            if(b == 2):
                count2 = count2 + 1
            if(b == 3):
                count3 = count3 + 1
            if(b == 4):
                count4 = count4 + 1

            if(c == 0):
                count0 = count0 + 1
            if(c == 1):
                count1 = count1 + 1
            if(c == 2):
                count2 = count2 + 1
            if(c == 3):
                count3 = count3 + 1
            if(c == 4):
                count4 = count4 + 1
            
            if(d == 0):
                count0 = count0 + 1
            if(d == 1):
                count1 = count1 + 1
            if(d == 2):
                count2 = count2 + 1
            if(d == 3):
                count3 = count3 + 1
            if(d == 4):
                count4 = count4 + 1

            if(e == 0):
                count0 = count0 + 1
            if(e == 1):
                count1 = count1 + 1
            if(e == 2):
                count2 = count2 + 1
            if(e == 3):
                count3 = count3 + 1
            if(e == 4):
                count4 = count4 + 1

            if(f == 0):
                count0 = count0 + 1
            if(f == 1):
                count1 = count1 + 1
            if(f == 2):
                count2 = count2 + 1
            if(f == 3):
                count3 = count3 + 1
            if(f == 4):
                count4 = count4 + 1

            if(count0 >= count0 and count0 >= count1 and count0 >= count2 and count0 >= count3 and count0 >= count4):
                return 0
            
            if(count1 >= count0 and count1 >= count1 and count1 >= count2 and count1 >= count3 and count1 >= count4):
                return 1

            if(count2 >= count0 and count2 >= count1 and count2 >= count2 and count2 >= count3 and count2 >= count4):
                return 2

            if(count3 >= count0 and count3 >= count1 and count3 >= count2 and count3 >= count3 and count3 >= count4):
                return 3

            if(count4 >= count0 and count4 >= count1 and count4 >= count2 and count4 >= count3 and count4 >= count4):
                return 4
                   
        
        #countlist = [count0,count1,count2,count3,count4]

        output = []
        for i in range(0,30000):
            x = majority(ANN_OUT[i],Y_DT[i],Y_KNN[i],Y_NB[i],Y_RF[i],Y_SVM[i])
            output.append(x)

        count0 = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0

        for i in range(0,30000):
            if(output[i] == 0):
                count0 = count0 + 1
            if(output[i] == 1):
                count1 = count1 + 1
            if(output[i] == 2):
                count2 = count2 + 1
            if(output[i] == 3):
                count3 = count3 + 1
            if(output[i] == 4):
                count4 = count4 + 1
            

        
        return render_template('output.html',msg1=str(count0),msg2=str(count1),msg3=str(count2),msg4=str(count3),msg5=str(count4),
                                msg1_ANN=str(count0_ANN),msg2_ANN=str(count1_ANN),msg3_ANN=str(count2_ANN),msg4_ANN=str(count3_ANN),msg5_ANN=str(count4_ANN),
                                msg1_DT=str(count0_DT),msg2_DT=str(count1_DT),msg3_DT=str(count2_DT),msg4_DT=str(count3_DT),msg5_DT=str(count4_DT),
                                msg1_KNN=str(count0_KNN),msg2_KNN=str(count1_KNN),msg3_KNN=str(count2_KNN),msg4_KNN=str(count3_KNN),msg5_KNN=str(count4_KNN),
                                msg1_NB=str(count0_NB),msg2_NB=str(count1_NB),msg3_NB=str(count2_NB),msg4_NB=str(count3_NB),msg5_NB=str(count4_NB),
                                msg1_RF=str(count0_RF),msg2_RF=str(count1_RF),msg3_RF=str(count2_RF),msg4_RF=str(count3_RF),msg5_RF=str(count4_RF),
                                msg1_SVM=str(count0_SVM),msg2_SVM=str(count1_SVM),msg3_SVM=str(count2_SVM),msg4_SVM=str(count3_SVM),msg5_SVM=str(count4_SVM))

if __name__ == '__main__':
    app.run()