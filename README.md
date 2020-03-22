# EstimateCosineFunction
Estimate Cosine Function with Neural Network

y = cos(X) where X>=-180 and X<=180 degree

## Dataset:
Training -  X_train contains randomized values with repeated selections within [-180,180] 
            y_train contains cos(X_train) + random noise within +/- 10% of cos(X_train)            
            
Test -      X_test contains values within the range of [-180,180] 
            y_test contains cos(X_test)
            
options:    N - number of available data points in the training set (default = 10000)
            theta_resolution - resolution of X (i.e. theta) (default = 0.01)

## Model Evaluation: 
- observe training/validation curves
- calculate test MSE adn RMSE
- observe any residual trends with the Bland-Altman plot

## Two main aprroaches:

### 1) Assume that only 1 'X' is available at inference time

#### EstimateCosine_simpleFC.py
- Fixed Architecture: Two hidden FC layers, each with 16 nodes
- Use early stopping to stop training when validation loss does not decrease for 5 epochs (avoid overfitting)
- Train-val curves
![alt text](https://github.com/NichaA/EstimateCosineFunction/raw/master/image/A1-1_trainval.png)
- Test MSE = 0.000018, Test RMSE = 0.01342
- Bland-Altman plot shows mean +/- SD = 0.00695 +/- 0.01148, 
    97.5% of the residuals are between -0.01556 and 0.02945, 
    the  model didn't perform well on the two edges (we will see if RandomizedSearchCV can optmize this)
![alt text](https://github.com/NichaA/EstimateCosineFunction/raw/master/image/A1-1_blandaltman.png)

#### EstimateCosine_RandomizedSearchCV.py
- Flexible FC Architectures for hyperparameter optimization with 3-fold cross-validation
- Parameter space for optimization include:
    - Number of adddtional hidden FC layers: hidden_layers = [1, 2, 4, 8]
    - Number of nodes in each FC layers: neurons = [16, 32, 64]
    - L2 regularization strength: reg_lambda = [0.1, 0.01, 0.001]
    - Batch size: batch_size = [100, 200, 400]
    - Number of epochs: epochs = [50, 100]
    Note that this includes 4 x 3 x 3 x 3 x 2 = 216 conditions. Let's randomly test 50 conditions (~23%).
- Randomized search results
Best: -0.001759 using {'reg_lambda': 0.001, 'neurons': 16, 'hidden_layers': 4, 'epochs': 100, 'batch_size': 400}
- Test MSE = 0.00014, Test RMSE = 0.011832
- Bland-Altman plot shows mean+/- SD = -0.00785 +/- 0.00874, 
    97.5% of the residuals are between -0.02947 and 0.00928,    
    the extreme ends of theta (at -180, and 180 degrees) has improved compared to previous fixed architecture,
    The distribution of the residuals is narrower here.
![alt text](https://github.com/NichaA/EstimateCosineFunction/raw/master/image/A1-2_blandaltman.png)

### 2) Assume that X is a time-series variable and a sequence of X's is available at inference time 
- LSTM model which take into account a sequence of 15 X values as input
- During training, use x_resolution=0.01. During testing, use x_resolution=0.013 to minimize overlapping 
- Use early stopping to stop training when validation loss does not decrease for 10 epochs (avoid overfitting)
- Train-val curves
![alt text](https://github.com/NichaA/EstimateCosineFunction/raw/master/image/A2-1_trainval.png)
- Test MSE = 0.00027, Test RMSE = 0.01654
- Bland-Altman plot shows mean +/- SD = -0.00033 +/- 0.01654, 
    97.5% of the residuals are between -0.03274 and 0.03208, 
![alt text](https://github.com/NichaA/EstimateCosineFunction/raw/master/image/A2-1_blandaltman.png)
