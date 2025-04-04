from customtkinter import *
import customtkinter
import threading
import tkinter




class MyWindow:
    def __init__(self, masterx, mastery):


        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")

        self.root = customtkinter.CTk()
        self.root.geometry(f"{masterx}x{mastery}")
        self.root.resizable(width=True, height=True)
        self.root.after(201, lambda :self.root.iconbitmap(r"neural.ico"))
        self.root.title("Transportation's neural network")
        self.PredictThread = None 

        self.MainCan = tkinter.Canvas(self.root, bg="#1a1a1a", highlightthickness=0)
        
        self.frame = customtkinter.CTkFrame(self.MainCan)
        self.features_frame = customtkinter.CTkFrame(self.frame, width=700, height=500, border_width=5, border_color="#0f4761")
        self.prediction_frame = customtkinter.CTkFrame(self.frame, width=350, height=500, border_width=5, border_color="#dbd70a")
        self.training_frame = customtkinter.CTkFrame(self.frame, border_width=5, border_color="#529949")
        self.Title = customtkinter.CTkLabel(self.frame, text="Transportation's traffic flow prediction \n using artifical neural network", anchor="center") 
        self.featuretitle = customtkinter.CTkLabel(self.features_frame, text="Input Features") 
        self.predictiontitle = customtkinter.CTkLabel(self.prediction_frame, text="Model Prediction")
        self.trainingtitle = customtkinter.CTkLabel(self.training_frame, text="Train Model")
        self.Timeofdayfeature = customtkinter.CTkLabel(self.features_frame, text="Time of day \n (in minutes)") 
        self.entry1 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.intervalfeature = customtkinter.CTkLabel(self.features_frame, text="Interval \n (No. every 4mins)") 
        self.entry2 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.speedfeature = customtkinter.CTkLabel(self.features_frame, text="Speed \n (miles)") 
        self.entry3 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.predictionbutton = customtkinter. CTkButton(self.features_frame, state="disabled", text="Predict", command=self.thread_predict, fg_color="#0f4761", font=("Arial (Body CS)", 38), border_width=2, border_color="#1a1a1a")
        self.trainingbutton = customtkinter. CTkButton(self.training_frame, state="disabled", text="Train", command=self.thread_train, fg_color="#529949", font=("Arial (Body CS)", 38), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        self.answer = customtkinter.CTkEntry(self.prediction_frame, state="disabled", fg_color="#8e8b06", border_color="#343638")
        self.checkbox = customtkinter.CTkCheckBox(self.frame, text = "Full screen", command=self.fullscreen)
        self.selectfilebutton=customtkinter.CTkButton(self.training_frame,text="Select dataset", command=self.selectdataset, fg_color="#529949", font=("Arial (Body CS)", 38), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        self.pathdataset=customtkinter.CTkEntry(self.training_frame, state="disabled", fg_color="#529949", border_width=2, border_color="#1a1a1a")
        self.Place(1680,780)
        self.MainCan.bind("<Configure>", self.OnResize)
        self.root.mainloop()

    def Place(self, masterx, mastery):
        self.MainCan.place(relx=0.5, rely=0.5, relheight=1, relwidth=1,anchor=CENTER)
        self.frame.place(relx=0.5, rely=0.5, anchor=CENTER,  relwidth=0.97619, relheight=0.9487179)
        self.features_frame.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER, relwidth=(700/1680), relheight=(500/780))
        self.prediction_frame.place(relx=0.85, rely=0.6, anchor=tkinter.CENTER, relwidth=(350/1680), relheight=(500/780))
        self.training_frame.place(relx=0.15, rely=0.6, anchor=tkinter.CENTER, relwidth=(350/1680), relheight=(500/780))

        self.Title.place(relx=0.5, rely=0.12, anchor=tkinter.CENTER)
        self.Title.configure(font=("Arial (Body CS)", (28*((masterx+mastery)/1680))))

        self.featuretitle.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)
        self.featuretitle.configure(font=("Arial (Body CS)", (20*((masterx+mastery)/1680))))

        self.predictiontitle.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)
        self.predictiontitle.configure(font=("Arial (Body CS)", (20*((masterx+mastery)/1680))))

        self.trainingtitle.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)
        self.trainingtitle.configure(font=("Arial (Body CS)", (20*((masterx+mastery)/1680))))

        self.Timeofdayfeature.place(relx=0.2, rely=0.25, anchor=tkinter.CENTER)
        self.Timeofdayfeature.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.entry1.place(relx=0.2, rely=0.55, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(250/780))
        self.entry1.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.intervalfeature.place(relx=0.5, rely=0.25, anchor=tkinter.CENTER)
        self.intervalfeature.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.entry2.place(relx=0.5, rely=0.55, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(250/780))
        self.entry2.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.speedfeature.place(relx=0.8, rely=0.25, anchor=tkinter.CENTER)
        self.speedfeature.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))
        
        self.entry3.place(relx=0.8, rely=0.55, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(250/780))
        self.entry3.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
        self.predictionbutton.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))), corner_radius=30)

        self.trainingbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(800/1680), relheight=(150/780))
        self.trainingbutton.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))), corner_radius=30)

        self.selectfilebutton.place(relx=0.5, rely=0.35, anchor=tkinter.CENTER, relwidth=(1500/1680), relheight=(150/780))
        self.selectfilebutton.configure(font=("Arial (Body CS)", (14*((masterx+mastery)/1680))), corner_radius=30)

        self.pathdataset.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER, relwidth=(1500/1680), relheight=(150/780))
        self.pathdataset.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.answer.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER, relwidth=(1000/1680), relheight=(250/780))
        self.answer.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.checkbox.place(relx=0.95, rely=0.96, anchor=CENTER)
        
        
    def selectdataset(self):
        global filename
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select csv file", filetypes=[("dataset files","*.csv"),])
        self.pathdataset.configure(state="normal")
        self.pathdataset.insert(0, filename)
        self.pathdataset.configure(state="disabled")
        self.entry1.configure(state="normal")
        self.entry2.configure(state="normal")
        self.entry3.configure(state="normal")
        self.trainingbutton.configure(state="normal")
        self.predictionbutton.configure(state="normal")



    def thread_predict(self):
        
        if self.PredictThread == None:
            self.PredictThread = threading.Thread(target=self.predict)
            self.PredictThread.start()
            return

        if self.PredictThread.is_alive() == False:
            self.PredictThread = threading.Thread(target=self.predict)
            self.PredictThread.start()
            return
        
    def thread_train(self):
        
        if self.PredictThread == None:
            self.PredictThread = threading.Thread(target=self.train)
            self.PredictThread.start()
            return

        if self.PredictThread.is_alive() == False:
            self.PredictThread = threading.Thread(target=self.train)
            self.PredictThread.start()
            return
        
    def train(self):
        self.trainingbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(1000/1680), relheight=(150/780))
        self.trainingbutton.configure(text="Training", state="disabled")

        import pandas as pd
        import numpy as np
        import tensorflow as tf
        import keras_tuner as kt
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import AdamW
        from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
        from sklearn.model_selection import train_test_split, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        global mse,rmse,mae,r2
        global rmse_scores,mse_scores,mae_scores,r2_scores
        global X_val, y_val,model

        data = pd.read_csv(filename)
        input_features = ["Time Period Ending", "Time Interval", "Avg mph"]
        target = "Total Volume"
        X = data[input_features]
        y = data[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        def build_model(hp):
            model = Sequential()
            model.add(Dense(hp.Int('units_1', 64, 512, step=64), activation='relu', input_shape=(X.shape[1],)))
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
            for i in range(hp.Int('num_layers', 2, 5)):
                model.add(Dense(hp.Int(f'units_{i+2}', 32, 256, step=32), activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(hp.Float(f'dropout_{i+2}', 0.1, 0.5, step=0.1)))

            model.add(Dense(1))

            model.compile(
                optimizer=AdamW(learning_rate=hp.Float('learning_rate', 0.0001, 0.01, sampling='LOG')),
                loss='mse',
                metrics=['mae']
            )

            return model

        tuner = kt.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=20, 
            executions_per_trial=2, 
            directory='tuner_results',
            project_name='trafficflow_strength_tuning'
        )


        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


        tuner.search(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores, rmse_scores, mae_scores, r2_scores = [], [], [], []

        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test), 
                                verbose=1, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5), 
                                                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)])


            y_pred = model.predict(X_test).flatten()


            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)

        model.save("TrafficflowNN.keras")

        self.trainingbutton.destroy()
        
        self.resultgbutton = customtkinter. CTkButton(self.training_frame, text="Results", command=self.result, fg_color="#529949", font=("Arial (Body CS)", 28), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        self.graphbutton = customtkinter. CTkButton(self.training_frame, text="Graph", command=self.graph, fg_color="#529949", font=("Arial (Body CS)", 28), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        
        self.resultgbutton.place(relx=0.26, rely=0.85, anchor=tkinter.CENTER, relwidth=(800/1680), relheight=(150/780))
        self.graphbutton.place(relx=0.74, rely=0.85, anchor=tkinter.CENTER, relwidth=(800/1680), relheight=(150/780))


    def result(self):
        self.resultgbutton.configure(state="disabled")
        import numpy as np
        new_window = customtkinter.CTkToplevel(self.root, fg_color="#212121")
        new_window.title("Results")
        new_window.geometry("425x275")
        new_window.resizable(width=False, height=False)


        def close():
            new_window.destroy()
            self.resultgbutton.configure(state="normal")

 
        self.titlewindow = customtkinter.CTkLabel(new_window, text=f"Performance indicator metrics", anchor="center", font=("Arial (Body CS)", 20))
        self.titlewindow.pack(pady=10)

        self.r2 = customtkinter.CTkLabel(new_window, text=f"R² Score: {np.mean(r2_scores):.4f}", anchor="center",font=("Arial (Body CS)", 20))
        self.r2.pack(pady=10)
        
        self.rmse = customtkinter.CTkLabel(new_window, text=f"Root Mean Squared Error (RMSE): {np.mean(rmse_scores):.4f}", anchor="center", font=("Arial (Body CS)", 20))
        self.rmse.pack(pady=10)

        self.mae = customtkinter.CTkLabel(new_window, text=f"Mean Absolute Error (MAE): {np.mean(mae_scores):.4f}", anchor="center", font=("Arial (Body CS)", 20))
        self.mae.pack(pady=10)

        new_button = customtkinter.CTkButton(new_window, text="Close Window", command=close)
        new_button.pack(pady=10)
        
        def confirm():
            new_window.destroy()
            self.resultgbutton.configure(state="normal")

        new_window.protocol("WM_DELETE_WINDOW", confirm)




    def graph(self):
        self.graphbutton.configure(state="disabled")
        import matplotlib.pyplot as plt
        y_pred_final = model.predict(X_val).flatten()
        plt.figure(figsize=(8, 6))
        plt.scatter(y_val, y_pred_final, alpha=0.7, color='blue', label="Predicted vs. Actual")
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--', color='red', label="Perfect Prediction")
        plt.xlabel("Actual Traffic Flow")
        plt.ylabel("Predicted Traffic Flow")
        plt.title("Actual vs Predicted Traffic Flow")
        plt.legend()
        plt.grid(True)
        plt.show()
        self.graphbutton.configure(state="normal")



    def predict(self):
        self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(700/1680), relheight=(150/780))
        self.predictionbutton.configure(text="predicting",state="disabled")
        import pandas as pd
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        data = pd.read_csv(filename)
        input_features = ["Time Period Ending", "Time Interval", "Avg mph"]
        target = "Total Volume"

        X = data[input_features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        time_period = self.entry1.get()
        time_interval = self.entry2.get()
        avg_mph = self.entry3.get()


        _error = False

        try:
            self.entry1.configure(fg_color="#ff0000")
            float(time_period)
            self.entry1.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry2.configure(fg_color="#ff0000")
            float(time_interval)
            self.entry2.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry3.configure(fg_color="#ff0000")  
            float(avg_mph)
            self.entry3.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        if _error:
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
            return
        
        
        custom_input = np.array([[time_period, time_interval, avg_mph]])
        custom_input_scaled = scaler.transform(custom_input)
        
        loadmodel = load_model("TrafficflowNN.keras")
        predicted_value = loadmodel.predict(custom_input_scaled)[0][0]
        self.answer.configure(state="normal")
        self.answer.delete(0, 'end')
        self.answer.insert(0, predicted_value)
        self.answer.configure(state="disabled")
        self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
        self.predictionbutton.configure(text="predict",state="normal")
        
    def fullscreen(self):
        if self.checkbox.get():
            self.root.attributes("-fullscreen", True)
        else:
            self.root.attributes("-fullscreen", False)

    def OnResize(self, event):
        self.Place(event.width, event.height)



MyWindow(1680, 780)