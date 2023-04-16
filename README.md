# PredictionIran-sPriceHouses

By posting different ads for real estate and land, the dataset is formed little by little

![image](https://user-images.githubusercontent.com/83599883/232286704-d032e8bb-81bd-49d7-9f5b-741b955605d9.png)

First, according to the entered neighborhood, we get the date and price of the desired location from the database

![image](https://user-images.githubusercontent.com/83599883/232288148-4e466308-7346-4944-8a70-852180602928.png)

Then we fill in its null values and normalize the data.
look_back, which is the number of previous time steps to use as input variables for predicting the next time period – in this case, the default is 1.

This assumption creates a data set where X is the number of prices at a given time (t) and Y is the number of prices at the next time (t + 1).

![image](https://user-images.githubusercontent.com/83599883/232288312-f66219dc-2e5e-4fc0-b5ff-4a7ded0aff35.png)

The network has a visible layer with 1 input, a hidden layer with 50 LSTM blocks or neurons, and an output layer that predicts a single value.
Default sigmoid activation function is used for LSTM blocks. The network is trained for 50 epochs and a batch size of 32 is used.
We reverse the predictions to ensure that the price returns in the same units as the original data in the API output.

![image](https://user-images.githubusercontent.com/83599883/232288360-b5c8731f-9a27-4a2b-87b8-b65721dda40a.png)
