import pandas as pd
import os
import joblib

from joblib import load
from sklearn.ensemble import RandomForestClassifier
import pickle


def prediction(gold_df, passengers_to_predict):
    # take raw data and fill in missing values. Then store it as the silver data
    # Let's do Embarked first. We can see that there are only 2 missing values in the Embarked column.
    # We can fill these missing values with the most common value in the column.
    
    
    # model_path = os.path.join(os.path.dirname(__file__), 'models_and_scalars/survival_model.joblib')
    
    # # Get the rows corresponding to the passengers_to_predict
    # passengers_df = gold_df[gold_df['PassengerId'].isin(passengers_to_predict)]
    # passengers_df = pd.get_dummies(passengers_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'LastName', 'FirstName', 'AdditionalName'], axis=1), drop_first=True)




    # # Load the model
    # model = load(model_path)
    
    # # List all features in passengers_df
    # print("Features in passengers_df:", passengers_df.columns.tolist())

    # # List all features in the model if it supports it
    # if hasattr(model, 'feature_importances_'):
    #     print("Features in model:", passengers_df.columns.tolist())
    # else:
    #     print("Model does not support feature importances.")
    
    # # Predict survival
    # survival_predictions = model.predict(passengers_df)
    # passengers_df['Survived'] = survival_predictions
    
    # return passengers_df
    train_df = gold_df
    # Drop rows where PassengerId is in the passengers_to_predict
    # Separate the rows where 'Survived' is 3 (to be predicted) from the rest
    train_df = gold_df[gold_df['Survived'] != 3]
    predict_df = gold_df[gold_df['Survived'] == 3]

    # Prepare the training data
    X_train = pd.get_dummies(train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'LastName', 'FirstName', 'AdditionalName'], axis=1), drop_first=True)
    y_train = train_df['Survived']

    # Initialize and train the model on the training dataset
    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    best_model.fit(X_train, y_train)

    # Save the trained model for deployment
    model_path = os.path.join(os.path.dirname(__file__), 'models_and_scalars/survival_model.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)

    # Prepare the data for prediction
    X_predict = pd.get_dummies(predict_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'LastName', 'FirstName', 'AdditionalName'], axis=1), drop_first=True)

    # Ensure the columns in X_predict match the columns used during training
    missing_cols = set(X_train.columns) - set(X_predict.columns)
    for col in missing_cols:
        X_predict[col] = 0
    X_predict = X_predict[X_train.columns]

    # Predict survival
    survival_predictions = best_model.predict(X_predict)
    predict_df['Survived'] = survival_predictions

    print(predict_df)

    return predict_df[['PassengerId', 'Survived']]
    
    
    print("Model and scaler saved successfully.")