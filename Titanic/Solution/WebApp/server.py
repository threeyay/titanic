from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import pandas as pd
import io
import os
from datetime import datetime


from data_processing.data_cleaning import clean_data
from data_processing.missing_data import fill_missing_data
from data_processing.prediction import prediction



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        socketio.emit('status', {'message': 'No file part'})
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        socketio.emit('status', {'message': 'No selected file'})
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        
        # Ensure the directory exists
        output_dir = 'data/raw_input'
        bronze_dir = 'data/data_bronze'
        silver_dir = 'data/data_silver'
        gold_dir   = 'data/data_gold'

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(bronze_dir, exist_ok=True)
        os.makedirs(silver_dir, exist_ok=True)
        os.makedirs(gold_dir, exist_ok=True)
        
        # Generate the output file name
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_filename = f"{os.path.splitext(file.filename)[0]}_{timestamp}.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Save the file
        df.to_csv(output_path, index=False)
        passenger_ids = df['PassengerId'].tolist()
        socketio.emit('status', {'message': f'Input File Saved'})
        
        # Perform data cleaning
        df = clean_data(output_path, bronze_dir)
        socketio.emit('status', {'message': f'Data Cleaning Complete. Saved to bronze layer'})
        
        # fill missing data
        df = fill_missing_data(bronze_dir, silver_dir, gold_dir)
        socketio.emit('status', {'message': f'Missing data filled. Saved to silver and gold folder'})
        
        df = prediction(df, passenger_ids)
        socketio.emit('status', {'message': f'Prediction complete.'})
        
        # Extract PassengerId and Survival columns
        result = df[['PassengerId', 'Survived']]

        # Convert the result to JSON
        result_json = result.to_dict(orient='records')

        # Return the result as JSON
        processed_data = jsonify({'result': result_json})
        print ('>><<>><<>>',processed_data)
        # Perform any processing on the DataFrame here
        
        # Perform any processing on the DataFrame here
        # For example, let's just return the DataFrame as JSON
        # processed_data = df.to_dict(orient='records')
        
        socketio.emit('status', {'message': 'Processing complete'})
        print (processed_data)
        return processed_data
    else:
        socketio.emit('status', {'message': 'Invalid file format'})
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    socketio.run(app, debug=True)