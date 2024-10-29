import pandas as pd
import os

def clean_data(input_file_path, output_dir):
    # take raw data and clean it up. Then store it as the bronze data
    print(">>>>>>>", input_file_path)
    titanic_df = pd.read_csv(input_file_path, na_filter=False)
    titanic_df = titanic_df.rename(columns={'Parch': 'ParCh'})

    # Ensure that the data types are correct and consistent
    titanic_df['PassengerId'] = titanic_df['PassengerId'].astype(int)
    titanic_df['Survived'] = pd.to_numeric(titanic_df['Survived'], errors='coerce').fillna(3).astype(int)
    titanic_df['Pclass'] = titanic_df['Pclass'].astype(int)
    titanic_df['Name'] = titanic_df['Name'].astype(str).fillna('')
    titanic_df['Sex'] = titanic_df['Sex'].astype(str).fillna('')
    titanic_df['Age'] = pd.to_numeric(titanic_df['Age'], errors='coerce')
    titanic_df['SibSp'] = titanic_df['SibSp'].astype(int)
    titanic_df['ParCh'] = titanic_df['ParCh'].astype(int)
    titanic_df['Ticket'] = titanic_df['Ticket'].astype(str).fillna('')
    titanic_df['Fare'] = pd.to_numeric(titanic_df['Fare'], errors='coerce')
    titanic_df['Cabin'] = titanic_df['Cabin'].astype(str).fillna('')
    titanic_df['Embarked'] = titanic_df['Embarked'].astype(str).fillna('')

    # it makes sense to also split the name column into salutation, first name and last name columns (so we can analyze if they are related to each other and how)
    titanic_df['Salutation'] = titanic_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    titanic_df['LastName'] = titanic_df['Name'].apply(lambda x: x.split(',')[0].strip())
    titanic_df['FirstName'] = titanic_df['Name'].apply(
        lambda x: x.split(',')[1].split('.')[1].split('(')[0].strip() if len(x.split(',')[1].split('.')) > 1 else ''
    )
    # Extract any additional names that are not captured as FirstName or LastName
    titanic_df['AdditionalName'] = titanic_df['Name'].apply(
        lambda x: x.split('(')[1].split(')')[0].strip() if '(' in x and ')' in x else ''
    )

    # Create a new column FamilySize which is the sum of SibSp and ParCh plus 1 for the passenger (so we can see if people travelling together had a higher chance of survival)
    titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['ParCh'] + 1
    
    # Create a new column Deck which is the first letter of the Cabin (so we can see if the deck had an impact on survival)    
    titanic_df['Deck'] = titanic_df['Cabin'].apply(lambda x: x[0] if x else '')

    
    # Write the cleaned data to the bronze folder
    file_name = os.path.basename(input_file_path)
    cleaned_file_name = output_dir+ '/'+ file_name
    titanic_df.to_csv(cleaned_file_name, index=False)
    
    return cleaned_file_name
