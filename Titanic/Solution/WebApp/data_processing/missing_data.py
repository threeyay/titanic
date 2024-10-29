import pandas as pd
import os
from joblib import load

def estimate_deck(deck_fare, pclass, per_person_fare, embarked, group_size):
    # Filter by Pclass and find the closest fare within the same Embarked and Group_Size
    possible_decks = deck_fare[(deck_fare['Pclass'] == pclass) &
                               (deck_fare['Embarked'] == embarked) &
                               (deck_fare['GroupSize'] == group_size)]
    
    # Sort by absolute difference in per_person_fare to find the closest match
    closest_deck = possible_decks.iloc[(possible_decks['PPFare'] - per_person_fare).abs().argsort()[:1]]
    deck =  closest_deck['Deck'].values[0] if not closest_deck.empty else ''

    if deck == '':
        possible_decks = deck_fare[(deck_fare['Pclass'] == pclass)]
        closest_deck = possible_decks.iloc[(possible_decks['PPFare'] - per_person_fare).abs().argsort()[:1]]
        deck = closest_deck['Deck'].values[0] if not closest_deck.empty else ''

    return deck


def fill_missing_data(input_dir, output_dir_oltp, output_dir_olap):
    # take raw data and fill in missing values. Then store it as the silver data
    # Let's do Embarked first. We can see that there are only 2 missing values in the Embarked column.
    # We can fill these missing values with the most common value in the column.
    
    # List all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    # Read each CSV file into a DataFrame and concatenate them into a single DataFrame
    titanic_df = pd.concat([pd.read_csv(os.path.join(input_dir, file)) for file in csv_files], ignore_index=True)
    
    
    # remove duplicate rows
    titanic_df.drop_duplicates(inplace=True)
    titanic_df_original = titanic_df.copy()

    # Find the most common value in the Embarked column
    most_common_embarked = titanic_df['Embarked'].mode()[0]
    # Fill null or empty strings in the 'Embarked' column with the most common value
    titanic_df['Embarked'] = titanic_df['Embarked'].replace('', most_common_embarked)
    
    
    feature_columns = ['Salutation', 'Sex', 'SibSp', 'ParCh']


    # Ensure categorical data is encoded; assume titanic_df has been prepared with dummies
    categorical_columns =   ['Salutation','Sex']
    # Given that most Salutations have few rows, we can consider consolidating them into fewer categories
    # For example, we can group all Salutations with less than 2 rows into an "Other" category
    salutation_counts = titanic_df['Salutation'].value_counts()
    other_salutations = salutation_counts[salutation_counts < 2].index
    titanic_df['Salutation'] = titanic_df['Salutation'].apply(lambda x: 'Other' if x in other_salutations else x)


    titanic_df = pd.get_dummies(titanic_df, columns=categorical_columns, drop_first=True)

    
    # Load the pre-trained model and scaler for age prediction
    model_path = os.path.join(os.path.dirname(__file__), 'models_and_scalars/age_model_RF.joblib')
    scaler_path = os.path.join(os.path.dirname(__file__), 'models_and_scalars/age_scaler.joblib')
    
    age_predictor = load(model_path)
    age_scaler = load(scaler_path)
    
    # Select features for age prediction
    feature_columns = ['SibSp', 'ParCh']
    # Add categorical columns by looping through columns and checking if it starts with a categorical column
    for col in titanic_df.columns:
        if any(col.startswith(cat_col) for cat_col in categorical_columns):
            feature_columns.append(col)
    X = titanic_df[feature_columns]

    # Predict missing ages
    missing_age_mask = titanic_df['Age'].isnull()
    titanic_df.loc[missing_age_mask, 'Age'] = age_predictor.predict(X[missing_age_mask])
    
    # Calculate the group size (number of passengers) for each ticket
    titanic_df['GroupSize'] = titanic_df.groupby('Ticket')['Ticket'].transform('count')

    # Calculate the per-person fare by dividing the Fare by the GroupSize for each ticket
    titanic_df['PPFare'] = titanic_df['Fare'] / titanic_df['GroupSize']
    
    deck_fare = titanic_df[titanic_df['Deck'] != ''].groupby(['Pclass', 'Deck', 'Embarked', 'GroupSize'])['PPFare'].median().reset_index()
    
    # Apply the function to assign Deck to rows with missing Deck information (empty string or NaN)
    titanic_df['Deck'] = titanic_df.apply(
        lambda row: estimate_deck(deck_fare, row['Pclass'], row['PPFare'], row['Embarked'], row['GroupSize']) 
        if row['Deck'] == '' or pd.isnull(row['Deck']) else row['Deck'],
        axis=1)
            
    titanic_df_original['Embarked'] = titanic_df['Embarked']
    titanic_df_original['Age'] = titanic_df['Age'].round(1)
    titanic_df_original['Age'] = titanic_df['Deck']
    titanic_df_original['PPFare'] = titanic_df['PPFare']
    titanic_df_original['GroupSize'] = titanic_df['GroupSize']
    
    # Write the cleaned data to the bronze folder
    file_name = 'titanic_silver.csv'
    cleaned_file_name = output_dir_oltp+ '/'+ file_name
    titanic_df_original.to_csv(cleaned_file_name, index=False, mode='w')

    # Write the cleaned data to the bronze folder
    file_name = 'titanic_gold.csv'
    cleaned_file_name = output_dir_olap+ '/'+ file_name
    titanic_df.to_csv(cleaned_file_name, index=False, mode='w')
    
    return titanic_df_original