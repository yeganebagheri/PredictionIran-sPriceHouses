import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
# from tensorflow.keras.layers import Dense, LSTM
# from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
import psycopg2
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)
from sqlalchemy import create_engine


app = FastAPI()
@app.get('/location/{name}')
def get_LocationPrice(name: str):
    student_height = []
    #newdf
    try:
        connection = psycopg2.connect(user="postgres",
                                    password="a12345",
                                    host="localhost",
                                    port="5432",
                                    database="RealEstate")
        cursor = connection.cursor()
        postgreSQL_select_Query = "SELECT date , price FROM post WHERE location = %s"
        #newdf = pd.read_sql('SELECT date , price FROM post WHERE location = %s',connection)
        
        cursor.execute(postgreSQL_select_Query,[name])
        print("Selecting rows from mobile table using cursor.fetchall")
        #mobile_records = cursor.fetchall()
        #print(mobile_records)
        engine = create_engine('postgresql://postgres:a12345@localhost:5432/RealEstate?sslmode=disable')
        # Connect to PostgreSQL server
        dbConnection= engine.connect();
        #query = ("SELECT date , price FROM post WHERE location = ? ")
        #location = "amirkabir"
        #data_df = pd.read_sql_query(query, engine, params=location)
        df = pd.read_sql(('select "date","price" from "post" '
                     'where "location" = %(dstart)s'),
                   dbConnection,params={"dstart":name})

        print(df)
    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


     
     #Now convert list into integer
    #for n in range(len(mobile_records)):
    #    student_height.append(mobile_records[n][1])
    #print(f"Heights are: {student_height}")


    #download the data
    #df = yf.download(tickers=['AAPL'], period='1y')
    #print(df)
    #y = df['Close'].fillna(method='ffill') #The fillna() method replaces the NULL values with a specified value
    y = df['price'].fillna(method='ffill')
    #ffill() is a synonym for the forward fill method.
    #print(y)
    #y = np.append(y)
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    caler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 4  # length of input sequences (lookback period)
    n_forecast = 2  # length of output sequences (forecast period)
    length = len(student_height)
    X = []
    Y = []
    #print(length)
    #print(y[1:61])
    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = X[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    #print(df)
    #df_past = df[['price']].reset_index()
    #print(df_past)
    df.rename(columns={'index': 'date', 'price': 'Actual'}, inplace=True)
    df_past = df
    print(df_past)
    #print(df_past['date'])
    df_past['date'] = pd.to_datetime(df_past['date'])
    #print(df_past['date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['date', 'Actual', 'Forecast'])
    df_future['date'] = pd.date_range(start=df_past['date'].iloc[-1] + pd.Timedelta(days=100), periods=n_forecast)
    fd = df_past['date'].iloc[-1]
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan

    #results = df_past.append(df_future).set_index('date')
    #df2 = df_future.to_json(orient = 'columns')
    df1 = df_future.fillna('')
    df2 = df1.values.tolist()
    # plot the results
    #results.plot(title='AAPL')
    

    return df2