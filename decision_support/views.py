from keras.models import load_model
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import csv
from datetime import datetime as dt_time
import plotly.graph_objects as go
import plotly.io as pio
from django.http import JsonResponse
from django.templatetags.static import static
from django.utils import timezone


def waterlvl_prediction():
    global forecast_values, original, data, df_forecast, forecast_values, last_known_value, forecast_dates
    global model_water, scaler, X_train, X_test, y_train, y_test

    model_water = load_model('Model_water.h5')

    first_data = pd.read_csv('water_data.csv')
    first_data['Rainfall'] = pd.to_numeric(first_data['Rainfall'], errors='coerce')

    # training and testing sets
    train_size = int(len(first_data) * 0.8)
    train_data = first_data.iloc[:train_size]

    # mean from training data
    water_mean = train_data['Water Level'].mean()
    drawdown_mean = train_data['Drawdown'].mean()

    # Fill missing values with means
    data = first_data.fillna(value={'Water Level': water_mean, 'Rainfall': 0, 'Drawdown': drawdown_mean}).copy()

    # Convert 'Date' column to datetime and set as index
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
    data.set_index('Date', inplace=True)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_length):
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, :])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)

    seq_length = 30
    X_train, y_train = create_sequences(scaled_data[:train_size], seq_length)
    X_test, y_test = create_sequences(scaled_data[train_size:], seq_length)



    train_dates = list(data.index)
    n_past = 15
    n_days_for_prediction= 480
    predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='d').tolist()   
    prediction = model_water.predict(X_test[-n_days_for_prediction:]) 
    prediction_copies = np.repeat(prediction, data.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]
    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())

    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Water Level':y_pred_future})

    df_forecast.set_index('Date', inplace=True)


    # past data
    original = data[['Water Level']]
    original = original.loc[(original.index >= original.index[-7]) & (original.index <= original.index[-1])]

    # forecasted data plot
    last_known_date = original.index[-1]
    start_date = last_known_date + pd.Timedelta(days=-6)
    forecast_end_date = start_date + pd.Timedelta(days=30)
    forecast_dates = pd.date_range(start=start_date, end=forecast_end_date) 
    forecast_values = df_forecast.loc[forecast_dates, 'Water Level']
    last_known_value = original['Water Level'].iloc[-1]

waterlvl_prediction()
now = datetime.datetime.now()
dateToday = now.strftime("%A %d %B, %Y  %I:%M%p")

@login_required(login_url='/admin/login/')
def Dashboard(request):
    forecasted_tom =  forecast_values.iloc[7]
    Yesterday = original['Water Level'].iloc[-2]

    last_date = data.index[-1]
    
    last_year_date = last_date.replace(year=last_date.year - 1)
    last_year_timestamp = pd.Timestamp(last_year_date)

    last_year_value = data.loc[last_year_timestamp, 'Water Level']
    date_last_year = last_year_timestamp.strftime("%B %d, %Y")
    min_water_level = data['Water Level'].min()


    def water_level_dashboard():

        # past data
        original = data[['Water Level']]
        original = original.loc[(original.index >= original.index[-7]) & (original.index <= original.index[-1])]

        # forecasted data plot
        last_known_date = original.index[-1]
        start_date = last_known_date + pd.Timedelta(days=-6)
        forecast_end_date = start_date + pd.Timedelta(days=7)
        forecast_dates = pd.date_range(start=start_date, end=forecast_end_date)
        forecast_values = df_forecast.loc[forecast_dates, 'Water Level']

        config = {'displaylogo': False, 'displayModeBar': True}

        past_trace = go.Scatter(
            x=original.index, 
            y=original['Water Level'],
            mode='markers+lines',
            marker=dict(color='#7CFC00', size=5),
            line=dict(width=1.5),
            name='Actual',
            hovertemplate='%{y:.2f}',  
        )

        forecast_trace = go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='markers+lines',
            marker=dict(color='orange', size=5),
            line=dict(width=1.5),
            name='Forecasted',
            hovertemplate='%{y:.2f}',
        )

        fig = go.Figure()
        fig.add_trace(past_trace)
        fig.add_trace(forecast_trace)

        fig.update_layout(
            xaxis=dict(
                title='Date',
                titlefont=dict(size=14, color='white', family='Helvetica'),
                tickformat='%b %d, %Y',
                tickangle=0,
                tickfont=dict(size=11, color='white', family='Helvetica')
            ),
            yaxis=dict(
                title='Water Level (m)',
                titlefont=dict(size=15, color='white', family='Helvetica'),
                tickfont=dict(size=11, color='white', family='Helvetica')
            ),
            margin=dict(t=0, l=65, b=70, r=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Helvetica',
                size=14,
                color='white'
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.08,
                xanchor='left',
                x=0
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(0, 0, 0, 0.7)',  
                font=dict(size=15, family='Helvetica', color='white')
            ),
            width = 550,
            height = 450,
            modebar_remove=['zoom', 'lasso','select2d','lasso2d','resetScale2d']
        )

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.2)', showspikes = True, spikecolor="white", spikethickness = 0.7, spikedash='solid', )
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.2)')

        html_str = pio.to_html(fig, config=config)
        return html_str

    plot = water_level_dashboard()


    def minimum_water_level(csv_file):
        min_water_level = float('inf')
        min_water_level_date = None

        with open(csv_file, 'r', newline='') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                water_level_str = row['Water Level']
                if water_level_str:  
                    try:
                        water_level = float(water_level_str)
                        if water_level < min_water_level:
                            min_water_level = water_level
                            min_water_level_date = row['Date']
                    except ValueError:
                        pass
            return min_water_level, min_water_level_date

    csv_file = 'water_data.csv' 

    min_water_level, min_water_level_date = minimum_water_level(csv_file)
    min_year_date = datetime.datetime.strptime(min_water_level_date, "%d-%b-%y").date()
    min_year_timestamp = pd.Timestamp(min_year_date)
    min_year_value = data.loc[min_year_timestamp, 'Water Level']
    min_water_level_date = min_year_timestamp.strftime("%B %d, %Y")

    def water_alloc():
        data = pd.read_csv('manila_water_data.csv')
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
        filtered_data = data.tail(6)
        filtered_data.set_index('Business Zone', inplace=True)
        filtered_data.sort_index(inplace=True)
        filtered_data['nrwv'] = filtered_data['Supply Volume'] - filtered_data['Bill Volume']
        y = list(range(len(filtered_data)))
        fig = go.Figure(data=[
            go.Bar(y=y, x=filtered_data['Supply Volume'], orientation='h', name="Supply Volume", base=0),
            go.Bar(y=y, x=-filtered_data['nrwv'], orientation='h', name="NRWV", base=0)
        ])
        config = {'displaylogo': False, 'displayModeBar': True}
        fig.update_layout(
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family="Arial, sans-serif",
                size=14, 
                color="white"  
            ),
            title=dict(
                text='Water Supply and NRWV',
                font=dict(
                    size=20,  
                    color="white"
                    ),
                x=0.5,
                xanchor= 'center'
            ),
            xaxis=dict(
                title=dict(
                    text='Supply Volume',
                    font=dict(
                        size=16,  
                        color="white"  
                    )
                ),
                tickfont=dict(
                    size=12,  
                    color="white"  
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Business Zone',
                    font=dict(
                        size=16,  
                        color="white" 
                    )
                ),
                tickfont=dict(
                    size=12,  
                    color="white"  
                )
            ),
        modebar_remove=['zoom', 'lasso','select2d','lasso2d']
        )
        fig.update_yaxes(ticktext=filtered_data.index,tickvals=y)

        html_str = fig.to_html(config=config)
        return html_str
    
    water_alloc_plot = water_alloc()

    date_today = df_forecast.index[13]
    date_yest = df_forecast.index[14]
    date_tom = df_forecast.index[15]
    alloc_data = pd.read_csv('manila_water_data.csv')
    alloc_data['Date'] = pd.to_datetime(alloc_data['Date'], format='%d-%b-%y')
    last_date = alloc_data['Date'].iloc[-1]
    alloc_date_format = pd.to_datetime(last_date, format='%d-%b-%y')
    get_month = alloc_date_format.month
    get_year = alloc_date_format.year
    day = 1
    datetime_obj = dt_time(year=get_year,month=get_month, day=day)
    display_year = datetime_obj.strftime("%Y")
    display_month = datetime_obj.strftime("%B")
    last_alloc_date = f"{display_month} {display_year}"
    return render(request, 'Dashboard.html', 
                  {'room_name': "broadcast",
                   'Tomorrow': forecasted_tom, 
                   'Today': last_known_value, 
                   'Yesterday': Yesterday,
                   'last_year_today': last_year_value,
                   'date_last_year': date_last_year,
                   'min_water_level': min_water_level,
                   'min_water_level_date': min_water_level_date,
                   'Date': dateToday,
                   'date_today': date_today,
                   'date_yest': date_yest,
                   'date_tom': date_tom,
                   'plot': plot,
                   'last_alloc_date': last_alloc_date,
                   'water_alloc_plot': water_alloc_plot,})


@login_required(login_url='/admin/login/')
def Forecast(request):
    def water_level_plot():
        waterlvl_prediction()
        last_known_date = original.index[-1]
        start_date = last_known_date + pd.Timedelta(days=-6)
        forecast_end_date = start_date + pd.Timedelta(days=470)
        forecast_dates = pd.date_range(start=start_date, end=forecast_end_date) 
        forecast_values = df_forecast.loc[forecast_dates, 'Water Level']
        last_known_value = original['Water Level'].iloc[-1]

        config = {'displaylogo': False, 'displayModeBar': True}

        past_trace = go.Scatter(
            x=original.index, 
            y=original['Water Level'],
            mode='markers+lines',
            marker=dict(color='#7CFC00', size=5),
            line=dict(width=1.5),
            name='Actual',
            hovertemplate='%{y:.2f} m',  
        )

        forecast_trace = go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='markers+lines',
            marker=dict(color='orange', size=5),
            line=dict(width=1.5),
            name='Forecasted',
            hovertemplate='%{y:.2f} m'
        )

        fig = go.Figure()
        fig.add_trace(past_trace)
        fig.add_trace(forecast_trace)

        fig.update_layout(
            xaxis=dict(
                title='Date',
                titlefont=dict(size=14, color='white'),
                tickformat='%b %d, %Y',
                tickangle=0,
                tickfont=dict(size=12, color='white'),
                range=[forecast_dates[0], forecast_dates[30]]

            ),
            yaxis=dict(
                title='Water Level (m)',
                titlefont=dict(size=15, color='white'),
                tickfont=dict(size=12, color='white')
            ),
            margin=dict(t=10, l=100, b=10, r=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Arial',
                size=14,
                color='white'
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.08,
                xanchor='left',
                x=0
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(0, 0, 0, 0.7)',  
                font=dict(size=15, family='Helvetica', color='white')
            ),
            width = 990,
            height = 600,
            modebar_remove=['zoom', 'lasso','select2d','lasso2d']
        )

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.3)', showspikes = True, spikecolor="white", spikethickness = 0.7, spikedash='solid', )
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.3)')

        html_str = pio.to_html(fig, config=config)
        with open('waterlvl_plot.html', 'w', encoding='utf-8') as f:
            f.write(html_str)

        return html_str

    
    forecasted_date = df_forecast.index[15]
    forecasted = df_forecast['Water Level'].iloc[15]
    forecasted = round(forecasted, 2)

    water_plot = water_level_plot()
    # PREDICTION
    train_predictions = model_water.predict(X_train)
    test_predictions = model_water.predict(X_test)
    # SCALING FOR PREDICTION
    train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, X_train[:, -1, 1:]), axis=1))[:, 0]
    test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, X_test[:, -1, 1:]), axis=1))[:, 0]

    y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

    # sMAPE of actual data to forecasted data
    test_predictions = np.array(test_predictions)
    y_test_inv = np.array(y_test_inv)
    fore_error = abs(y_test_inv - test_predictions)
    fore_percentage_error = fore_error / (abs(y_test_inv) + abs(test_predictions))
    fore_smape = 100 * np.mean(fore_percentage_error)
    fore_smape = round(fore_smape,2)
    #fore_smape = np.mean((np.abs(test_predictions - y_test_inv) / np.abs(test_predictions + y_test_inv))) * 100
    #fore_smape = 100 - fore_smape

    # sMAPE of actual data to forecasted data
    numerator = abs(df_forecast['Water Level'].iloc[15] - original['Water Level'].iloc[-1])
    denominator = (abs(original['Water Level'].iloc[-1]) + abs(df_forecast['Water Level'].iloc[15]))
    act_percentage_water_error = numerator / denominator

    act_smape = 100 * act_percentage_water_error
    act_smape = 100 - act_smape
    act_smape = round(act_smape, 2)





    def rainfall_plot():
        model_rainfall = load_model('Model_rainfall.keras')
        first_data = pd.read_csv('rainfall_data.csv')
        first_data['RAINFALL'] = pd.to_numeric(first_data['RAINFALL'], errors='coerce')

        # Split data into training and testing sets
        train_size = int(len(first_data) * 0.8)
        train_data = first_data.iloc[:train_size]
        test_data = first_data.iloc[train_size:]

        tmax_mean = train_data['TMAX'].mean()
        tmin_mean = train_data['TMIN'].mean()
        tmean_mean = train_data['TMEAN'].mean()
        wind_speed_mean = train_data['WIND_SPEED'].mean()
        wind_direct_mean = train_data['WIND_DIRECTION'].mean()
        rh_mean = train_data['RH'].mean()

        data = first_data.fillna(value={'RAINFALL': 0, 'TMAX': tmax_mean, 'TMIN': tmin_mean, 'TMEAN': tmean_mean, 'WIND_SPEED': wind_speed_mean, 'WIND_DIRECTION': wind_direct_mean, 'RH': rh_mean}).copy()
        data['Date'] = pd.to_datetime(data[['YEAR', 'MONTH', 'DAY']], format='%d-%b-%y')
        data.set_index('Date', inplace=True)
        data.drop(columns=['YEAR', 'DAY', 'MONTH'], inplace=True)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        def create_sequences(data, seq_length):
            X = []
            y = []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length, :])
                y.append(data[i+seq_length, 0])
            return np.array(X), np.array(y)

        seq_length = 12
        X_train, y_train = create_sequences(scaled_data[:train_size], seq_length)
        X_test, y_test = create_sequences(scaled_data[train_size:], seq_length)

        train_dates = list(data.index)
        n_past = 10
        n_days_for_prediction= 300 #365
        predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='d').tolist()
        prediction = model_rainfall.predict(X_train[-n_days_for_prediction:]) 
        prediction_copies = np.repeat(prediction, data.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]
        forecast_dates = []
        for time_i in predict_period_dates:
            forecast_dates.append(time_i.date())
            
        df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'RAINFALL':y_pred_future})
        df_forecast.set_index('Date', inplace=True)

        # For the past data
        original = data[['RAINFALL']]
        original = original.loc[(original.index >= original.index[-10]) & (original.index <= original.index[-1])]

        # For the forecasted data plot
        last_known_date = original.index[-1]
        start_date = last_known_date + pd.Timedelta(days=-8)
        forecast_end_date = start_date + pd.Timedelta(days=35)
        forecast_dates = pd.date_range(start=start_date, end=forecast_end_date)
        forecast_values_rain = df_forecast.loc[forecast_dates, 'RAINFALL']

        config = {'displaylogo': False, 'displayModeBar': True}

        past_trace = go.Scatter(
            x=original.index, 
            y=original['RAINFALL'],
            mode='markers+lines',
            marker=dict(color='#7CFC00', size=5),
            line=dict(width=1.5),
            name='Actual',
            hovertemplate='%{y:.2f} mm',  
        )

        forecast_trace = go.Scatter(
            x=forecast_dates,
            y=forecast_values_rain,
            mode='markers+lines',
            marker=dict(color='orange', size=5),
            line=dict(width=1.5),
            name='Forecasted',
            hovertemplate='%{y:.2f} mm',
        )

        fig = go.Figure()
        fig.add_trace(past_trace)
        fig.add_trace(forecast_trace)

        fig.update_layout(
            xaxis=dict(
                title='Date',
                titlefont=dict(size=14, color='white'),
                tickformat='%b %d, %Y',
                tickangle=0,
                tickfont=dict(size=12, color='white')
            ),
            yaxis=dict(
                title='Rainfall (mm)',
                titlefont=dict(size=15, color='white'),
                tickfont=dict(size=12, color='white')
            ),
            margin=dict(t=0, l=100, b=10, r=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Arial',
                size=14,
                color='white'
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.08,
                xanchor='left',
                x=0
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(0, 0, 0, 0.7)', 
                font=dict(size=15, family='Helvetica', color='white') 
            ),
            width = 990,
            height = 600,
            modebar_remove=['zoom', 'lasso','select2d','lasso2d','resetScale2d']
        )

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.3)', showspikes = True, spikecolor="white", spikethickness = 0.7, spikedash='solid', )
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.3)')

        html_str = pio.to_html(fig, config=config)
        with open('rainfall_plot.html', 'w', encoding='utf-8') as f:
            f.write(html_str)


            
        # PREDICTION
        train_predictions = model_rainfall.predict(X_train)
        test_predictions = model_rainfall.predict(X_test)
        # SCALING FOR PREDICTION
        train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, X_train[:, -1, 1:]), axis=1))[:, 0]
        test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, X_test[:, -1, 1:]), axis=1))[:, 0]

        y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

        # sMAPE of actual data to forecasted data
        test_predictions = np.array(test_predictions)
        y_test_inv = np.array(y_test_inv)


        # SMAPE of actual data 
        numerator = abs(df_forecast['RAINFALL'].iloc[8] - original['RAINFALL'].iloc[-1])
        denominator = (abs(original['RAINFALL'].iloc[-1]) + abs(df_forecast['RAINFALL'].iloc[8]))
        act_percentage_rain_error = numerator / denominator

        act_rain_smape = 100 * act_percentage_rain_error
        act_rain_smape = 100 - act_rain_smape
        act_rain_smape = round(act_rain_smape, 2)


        forecast_rain = df_forecast['RAINFALL'].iloc[8]
        forecast_rain = round(forecast_rain,2)

        actual_rain = original['RAINFALL'].iloc[-1]
        

        return fore_rain_smape, act_rain_smape, forecast_rain, actual_rain, html_str

    forecast_rain = 0
    actual_rain = 0
    fore_rain_smape = 0.00
    act_rain_smape = 0.00


    def drawdown_plot():
        model_drawdown = load_model('Model_drawdown.h5')
        first_data = pd.read_csv('water_data.csv')
        columns = ['Date', 'Drawdown', 'Rainfall', 'Water Level'] 
        first_data = first_data[columns]
        first_data['Rainfall'] = pd.to_numeric(first_data['Rainfall'], errors='coerce')

        # Split data into training and testing sets
        train_size = int(len(first_data) * 0.8)
        train_data = first_data.iloc[:train_size]
        test_data = first_data.iloc[train_size:]

        # Calculate means from training data
        water_mean = train_data['Water Level'].mean()
        drawdown_mean = train_data['Drawdown'].mean()

        # Fill missing values with means
        data = first_data.fillna(value={'Drawdown': drawdown_mean, 'Rainfall': 0, 'Water Level': water_mean,}).copy()

        # Convert 'Date' column to datetime and set as index
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
        data.set_index('Date', inplace=True)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        def create_sequences(data, seq_length):
            X = []
            y = []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length, :])
                y.append(data[i+seq_length, 0])
            return np.array(X), np.array(y)

        seq_length = 10
        X_train, y_train = create_sequences(scaled_data[:train_size], seq_length)
        X_test, y_test = create_sequences(scaled_data[train_size:], seq_length)



        train_dates = list(data.index)
        n_past = 89
        n_days_for_prediction= 221
        
        predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='d').tolist()
        prediction = model_drawdown.predict(X_test[-n_days_for_prediction:]) 
        prediction_copies = np.repeat(prediction, data.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]
        forecast_dates = []
        for time_i in predict_period_dates:
            forecast_dates.append(time_i.date())
            
        df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Drawdown':y_pred_future})
        df_forecast.set_index('Date', inplace=True)
        original = data[['Drawdown']]
        actual_drawdown = original['Drawdown'].iloc[-1]
        original = original.loc[(original.index >= original.index[-7]) & (original.index <= original.index[-1])]

        # For the forecasted data plot
        last_known_date = original.index[-1]
        start_date = last_known_date + pd.Timedelta(days=-6)
        forecast_end_date = start_date + pd.Timedelta(days=30)
        forecast_dates = pd.date_range(start=start_date, end=forecast_end_date)
        forecast_values_drawdwn = df_forecast.loc[forecast_dates, 'Drawdown']

        config = {'displaylogo': False, 'displayModeBar': True}

        past_trace = go.Scatter(
            x=original.index, 
            y=original['Drawdown'],
            mode='markers+lines',
            marker=dict(color='#7CFC00', size=5),
            line=dict(width=1.5),
            name='Actual',
            hovertemplate='%{y:.2f} cu m',  
        )

        forecast_trace = go.Scatter(
            x=forecast_dates,
            y=forecast_values_drawdwn,
            mode='markers+lines',
            marker=dict(color='orange', size=5),
            line=dict(width=1.5),
            name='Forecasted',
            hovertemplate='%{y:.2f} cu m',
        )

        fig = go.Figure()
        fig.add_trace(past_trace)
        fig.add_trace(forecast_trace)

        fig.update_layout(
            xaxis=dict(
                title='Date',
                titlefont=dict(size=14, color='white'),
                tickformat='%b %d, %Y',
                tickangle=0,
                tickfont=dict(size=12, color='white')
            ),
            yaxis=dict(
                title='Drawdown (cu m)',
                titlefont=dict(size=15, color='white'),
                tickfont=dict(size=12, color='white')
            ),
            margin=dict(t=0, l=100, b=10, r=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Arial',
                size=14,
                color='white'
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.08,
                xanchor='left',
                x=0
            ),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='rgba(0, 0, 0, 0.7)', 
                font=dict(size=15, family='Helvetica', color='white') 
            ),
            width = 990,
            height = 600,
            modebar_remove=['zoom', 'lasso','select2d','lasso2d','resetScale2d']
        )

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.3)', showspikes = True, spikecolor="white", spikethickness = 0.7, spikedash='solid', )
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255, 255, 255, 0.2)')

        html_str = pio.to_html(fig, config=config)
        with open('drawdown_plot.html', 'w', encoding='utf-8') as f:
            f.write(html_str)
         # PREDICTION
        train_predictions = model_drawdown.predict(X_train)
        test_predictions = model_drawdown.predict(X_test)
        # SCALING FOR PREDICTION
        train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, X_train[:, -1, 1:]), axis=1))[:, 0]
        test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, X_test[:, -1, 1:]), axis=1))[:, 0]

        y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

        # sMAPE of actual data to forecasted data
        test_predictions = np.array(test_predictions)
        y_test_inv = np.array(y_test_inv)
        fore_drawdown_smape = np.mean((np.abs(test_predictions - y_test_inv) / np.abs(test_predictions + y_test_inv)/2)) * 100
        #fore_drawdown_smape = 200 - fore_drawdown_smape
        fore_drawdown_smape = round(fore_drawdown_smape,2)

        # sMAPE of actual data to forecasted data
        numerator = abs(df_forecast['Drawdown'].iloc[89] - original['Drawdown'].iloc[-1])
        denominator = (abs(original['Drawdown'].iloc[-1]) + abs(df_forecast['Drawdown'].iloc[89]))
        act_percentage_drawdown_error = numerator / denominator

        act_drawdown_smape = 100 * act_percentage_drawdown_error
        act_drawdown_smape = 100 - act_drawdown_smape
        act_drawdown_smape = round(act_drawdown_smape, 2)
        forecast_drawdown = df_forecast['Drawdown'].iloc[89]
        return fore_drawdown_smape, act_drawdown_smape, forecast_drawdown, actual_drawdown, html_str
    
    forecast_drawdown = 0
    actual_drawdown = 0
    fore_drawdown_smape = 0.00
    act_drawdown_smape = 0.00

    

    forecast_all = request.GET.get('forecast_all', None)
    forecast_waterlvl = request.GET.get('forecast_waterlvl', None)
    forecast_rainfall = request.GET.get('forecast_rainfall', None)
    forecast_drawdwn = request.GET.get('forecast_drawdown', None)
    if forecast_all:
        water_plot = water_level_plot()
        fore_rain_smape, act_rain_smape, forecast_rain, actual_rain, rain_plot = rainfall_plot()
        fore_drawdown_smape, act_drawdown_smape, forecast_drawdown, actual_drawdown, drawdown_interact_plot = drawdown_plot()
    elif forecast_waterlvl:
        water_plot = water_level_plot()
        with open('rainfall_plot.html', 'r', encoding='utf-8') as f:
            rain_plot = f.read()
        with open('drawdown_plot.html', 'r', encoding='utf-8') as f:
            drawdown_interact_plot = f.read()
    elif forecast_rainfall:
        fore_rain_smape, act_rain_smape, forecast_rain, actual_rain, rain_plot = rainfall_plot()
        with open('drawdown_plot.html', 'r', encoding='utf-8') as f:
            drawdown_interact_plot = f.read()
    elif forecast_drawdwn:
        fore_drawdown_smape, act_drawdown_smape, forecast_drawdown, actual_drawdown, drawdown_interact_plot = drawdown_plot()
        with open('rainfall_plot.html', 'r', encoding='utf-8') as f:
            rain_plot = f.read()
    else:
        with open('waterlvl_plot.html', 'r', encoding='utf-8') as f:
            water_plot = f.read()
        with open('rainfall_plot.html', 'r', encoding='utf-8') as f:
            rain_plot = f.read()
        with open('drawdown_plot.html', 'r', encoding='utf-8') as f:
            drawdown_interact_plot = f.read()

    with open('water_level_test_set.html', 'r', encoding='utf-8') as f:
        water_level_test_set = f.read()
    with open('rainfall_test_set.html', 'r', encoding='utf-8') as f:
        rainfall_test_set = f.read()
    with open('drawdown_test_set.html', 'r', encoding='utf-8') as f:
        drawdown_test_set = f.read()
    
    
                

  



    return render(request, 'Forecast.html', 
                  {'Date': dateToday,
                   'actual': last_known_value,
                   'forecasted': forecasted,
                   'forecasted_date': forecasted_date,
                   'fore_smape': fore_smape,
                   'act_smape': act_smape,
                   'forecast_drawdown': forecast_drawdown,
                   'actual_drawdown': actual_drawdown,
                   'fore_drawdown_smape':fore_drawdown_smape,
                   'act_drawdown_smape': act_drawdown_smape,
                   'actual_rain': actual_rain,
                   'forecast_rain': forecast_rain,
                   'fore_rain_smape': fore_rain_smape,
                   'act_rain_smape': act_rain_smape,
                   'water_plot': water_plot,
                   'rain_plot': rain_plot,
                   'drawdown_interact_plot': drawdown_interact_plot,
                   'water_level_test_set': water_level_test_set,
                   'rainfall_test_set':rainfall_test_set,
                   'drawdown_test_set': drawdown_test_set})
@login_required(login_url='/admin/login/')
def Business_zone (request):
    global filtered_data
    data = pd.read_csv('manila_water_data.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
    last_date = data['Date'].iloc[-1]
    alloc_date_format = pd.to_datetime(last_date, format='%d-%b-%y')
    get_month = alloc_date_format.month
    get_year = alloc_date_format.year
    get_graph = 1
    day = 1
    if request.method == 'POST':
        get_month = int(request.POST['month']) 
        get_year = int(request.POST['year'])
        get_graph = int(request.POST['graph'])

    datetime_obj = dt_time(year=get_year,month=get_month, day=day)

    display_year = datetime_obj.strftime("%Y")
    display_month = datetime_obj.strftime("%B")
    display_date = f"{display_month} {display_year}"
    month_date = display_date

    date_string = datetime_obj.strftime("%d-%b-%y")

    filtered_data = data[data['Date'].dt.strftime('%d-%b-%y') == date_string]
    filtered_data.set_index('Business Zone', inplace=True)
    filtered_data.sort_index(inplace=True)
    index = filtered_data.index
    filtered_data['nrwv'] = filtered_data['Supply Volume'] - filtered_data['Bill Volume']

    def bar_chart():
        fig = go.Figure()

        config = {'displaylogo': False, 'displayModeBar': True}
        fig.add_trace(go.Bar(x=index, y=filtered_data['Supply Volume'], name='Supply Volume', marker_color='blue'))
        fig.add_trace(go.Bar(x=index, y=filtered_data['nrwv'], name='NRWV', marker_color='skyblue'))

        fig.update_layout(
            title='BAR CHART',
            xaxis_title='Business Zone',
            yaxis_title='Volume',
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',  
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(
                family='Arial',
                size=14,
                color='white'
            ),
            width=1000,
            height=600,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            modebar_remove=['zoom', 'pan', 'lasso', 'pan2d','select2d','lasso2d','resetScale2d']
        )
        html_str = pio.to_html(fig, config=config)

        with open('bar_chart.html', 'w', encoding='utf-8') as f:
            f.write(html_str)

        plt.close() 
        return html_str


        
    def pie_chart():
        colors = ("orange", "cyan", "brown", "grey", "indigo", "beige")

        config = {'displaylogo': False, 'displayModeBar': True}

        def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.1f}%\n({:d} mld)".format(pct, absolute)

        fig = go.Figure()

        fig.add_trace(go.Pie(
            labels=index,
            values=filtered_data['nrwv'],
            textinfo='label+percent',
            textposition='inside',
            marker=dict(colors=colors),
            sort=False,
            domain={'x': [0.55, 1], 'y': [0.05, 0.95]}, 
            title='Non-Revenue Water Volume'
        ))

        fig.add_trace(go.Pie(
            labels=index,
            values=filtered_data['Supply Volume'],
            textinfo='label+percent',
            textposition='inside',
            marker=dict(colors=colors),
            sort=False,
            domain={'x': [0, 0.45], 'y': [0.05, 0.95]},
            title='Supply Volume'
        ))

        fig.update_layout(
            title='PIE CHART',
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',  
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Arial',
                size=14,
                color='white'
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=0,
                xanchor='center',
                x=0.5
            ),
            width=950,
            height=600,
            margin=dict(t=50, b=10, l=10, r=10)
        )
        
        html_str = pio.to_html(fig, config=config)

        with open('pie_chart.html', 'w', encoding='utf-8') as f:
            f.write(html_str)

        plt.close() 
        return html_str
    
    def line_chart():
        araneta = data[data['Business Zone'] == 'Araneta-Libis']
        elliptical = data[data['Business Zone'] == 'Elliptical']
        sjuan = data[data['Business Zone'] == 'San Juan']
        tandang_sora = data[data['Business Zone'] == 'Tandang sora']
        timog = data[data['Business Zone'] == 'Timog']
        up_katipunan = data[data['Business Zone'] == 'Up-Katipunan']

        fig = go.Figure()
        config = {'displaylogo': False, 'displayModeBar': True}
        fig.add_trace(go.Scatter(x=araneta['Date'], y=araneta['Supply Volume'], mode='lines+markers', name='Araneta-Libis'))
        fig.add_trace(go.Scatter(x=elliptical['Date'], y=elliptical['Supply Volume'], mode='lines+markers', name='Elliptical'))
        fig.add_trace(go.Scatter(x=sjuan['Date'], y=sjuan['Supply Volume'], mode='lines+markers', name='San Juan'))
        fig.add_trace(go.Scatter(x=tandang_sora['Date'], y=tandang_sora['Supply Volume'], mode='lines+markers', name='Tandang sora'))
        fig.add_trace(go.Scatter(x=timog['Date'], y=timog['Supply Volume'], mode='lines+markers', name='Timog'))
        fig.add_trace(go.Scatter(x=up_katipunan['Date'], y=up_katipunan['Supply Volume'], mode='lines+markers', name='Up Katipunan'))

        fig.update_layout(
            title='LINE CHART',
            xaxis_title='Date',
            yaxis_title='Supply Volume',
            plot_bgcolor='rgba(0,0,0,0)',  
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.5)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.5)', gridwidth=1),
            font=dict(
                family='Arial',
                size=14,
                color='white'
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.08,
                xanchor='right',
                x=1
            ),
            width = 980,
            height = 600,
            modebar_remove=['zoom', 'lasso','select2d','lasso2d','resetScale2d']
        )
        html_str = pio.to_html(fig, config=config)

        with open('line_chart.html', 'w', encoding='utf-8') as f:
            f.write(html_str)

        plt.close() 
        return html_str




    if get_graph == 2:
        chart = bar_chart()
    elif get_graph == 3:
        chart = pie_chart()
    elif get_graph == 4:
        chart = line_chart()
        display_date = 'Monthly'
    else:
        with open('bar_chart.html', 'r', encoding='utf-8') as f:
            chart = f.read()


    supply = filtered_data['Supply Volume'].sum()
    total_nrwv = filtered_data['nrwv'].sum()
    nrwv_percentage = (total_nrwv / supply) * 100
    total_supply = supply - total_nrwv

    def WaterAlloc(location):
        location_name = filtered_data.loc[location]
        supply = location_name['Supply Volume']
        bill = location_name['Bill Volume']
        nrwv = location_name['Supply Volume'] - location_name['Bill Volume'] 
        total_sv = location_name['Supply Volume'] - nrwv
        return supply, bill, nrwv, total_sv
    
    araneta_sv = 0
    araneta_bill = 0
    araneta_nrwv = 0
    araneta_ws = 0

    araneta_sv, araneta_bill, araneta_nrwv, araneta_ws = WaterAlloc('Araneta-Libis')

    elli_sv = 0
    elli_bill = 0
    elli_nrwv = 0
    elli_ws = 0

    elli_sv, elli_bill, elli_nrwv, elli_ws = WaterAlloc('Elliptical')

    sj_sv = 0
    sj_bill = 0
    sj_nrwv = 0
    sj_ws = 0

    sj_sv, sj_bill, sj_nrwv, sj_ws = WaterAlloc('San Juan')

    ts_sv = 0
    ts_bill = 0
    ts_nrwv = 0
    ts_ws = 0

    ts_sv, ts_bill, ts_nrwv, ts_ws = WaterAlloc('Tandang sora')

    timog_sv = 0
    timog_bill = 0
    timog_nrwv = 0
    timog_ws = 0

    timog_sv, timog_bill, timog_nrwv, timog_ws = WaterAlloc('Timog')
    
    up_sv = 0
    up_bill = 0
    up_nrwv = 0
    up_ws = 0

    up_sv, up_bill, up_nrwv, up_ws = WaterAlloc('Up-Katipunan')

    supply_volume = araneta_sv + elli_sv + sj_sv + ts_sv + timog_sv + up_sv
    bill_volume = araneta_bill + elli_bill + sj_bill + ts_bill + timog_bill + up_bill
    nrw_volume = araneta_nrwv + elli_nrwv + sj_nrwv + ts_nrwv + timog_nrwv + up_nrwv
    water_supply = araneta_ws + elli_ws + sj_ws + ts_ws + timog_ws + up_ws

    

    


    return render(request, 'Business-Zones.html', 
                {'Date': dateToday,
                'supply': supply,
                'total_supply': total_supply,
                'total_nrwv':total_nrwv,
                'nrwv_percentage': nrwv_percentage,
                'display_date':display_date,
                'month_date':month_date,
                'chart': chart,
                'araneta_sv': araneta_sv, 'araneta_bill': araneta_bill,'araneta_nrwv': araneta_nrwv, 'araneta_ws': araneta_ws,
                'elli_sv': elli_sv, 'elli_bill': elli_bill, 'elli_nrwv': elli_nrwv, 'elli_ws': elli_ws,
                'sj_sv':sj_sv, 'sj_bill': sj_bill, 'sj_nrwv': sj_nrwv, 'sj_ws': sj_ws,
                'ts_sv': ts_sv, 'ts_bill': ts_bill, 'ts_nrwv': ts_nrwv, 'ts_ws': ts_ws,
                'timog_sv': timog_sv, 'timog_bill': timog_bill, 'timog_nrwv': timog_nrwv, 'timog_ws': timog_ws,
                'up_sv': up_sv, 'up_bill': up_bill, 'up_nrwv': up_nrwv, 'up_ws': up_ws,
                'supply_volume': supply_volume,
                'bill_volume': bill_volume,
                'nrw_volume': nrw_volume,
                'water_supply': water_supply})
    
def Img_map(request):
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        if request.method == 'GET':
            df = pd.DataFrame(filtered_data)
            location = request.GET.get('location_name')
            if location in df.index:
                business_zone = df.loc[location]
                nrwv = business_zone['Supply Volume'] - business_zone['Bill Volume']
                bv = business_zone['Bill Volume']
                sv = business_zone['Supply Volume']
                nrwv_percentage = (nrwv / sv) * 100
                water_supply = sv - nrwv

                nrwv = round(nrwv, 2)
                sv = round(sv, 2)
                nrwv_percentage = round(nrwv_percentage, 2)

                img_src = ''
                if location == 'Elliptical':
                    img_src = 'img/bz-map(elliptical).png'
                elif location == 'Tandang sora':
                    img_src = 'img/bz-map(tsora).png'
                elif location == 'Timog':
                    img_src = 'img/bz-map(timog).png'
                elif location == 'Up-Katipunan':
                    img_src = 'img/bz-map(up).png'
                elif location == 'Araneta-Libis':
                    img_src = 'img/bz-map(araneta).png'
                elif location == 'San Juan':
                    img_src = 'img/bz-map(sjuan).png'

                data = {
                    'nrwv': nrwv,
                    'sv': sv,
                    'bv': bv,
                    'water_supply': water_supply,
                    'nrwv_percentage': nrwv_percentage,
                    'location': location,
                    'img_src': static(img_src) if img_src else ''
                }

                return JsonResponse(data)
            else:
                return JsonResponse({'error': 'Location not found'}, status=404)
    return JsonResponse({'error': 'Invalid request'}, status=400)



def Get_current_datetime(request):
    now = timezone.localtime(timezone.now())
    current_datetime = now.strftime("%A %d %B, %Y  %I:%M%p")
    return JsonResponse({'current_datetime': current_datetime})