import flask
from flask import Flask, request, jsonify
# Initialize the Flask app
app = Flask(__name__)
#
#Inference Wrapper starts here

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import xgboost as xgb
import joblib
import os
import json
from scipy import stats
from datetime import datetime

#Change parameters here
model_version = '4.0a'+'5' #IPD version
HISTORY_LENGTH = 10
scalerUsed = False
debugInputs = False #exports feature engr'd inputs
batchSkillUsed = False #turns off skill aggregator (uses item model directly)
skillTransformerUsed = False #sigmoid scales predictions (closer to 0 & 1)

#Load all artifacts (5): 2 models, 1 scaler, 1 score transformer, IP file

# Load the proficiency model
proficiency_model = xgb.Booster()
proficiency_model.load_model('proficiency_model.json')

# Load the confidence model
confidence_model = xgb.Booster()
confidence_model.load_model('confidence_model.json')

# Load the proficiency scaler
class NanPreservingScaler:
    def __init__(self):
        self.scalers = {}
        self.column_means = {}
        self.numeric_columns = None

    def partial_fit(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        
        if self.numeric_columns is None:
            self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        X_numeric = X[self.numeric_columns]
        
        for col in self.numeric_columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
            
            col_data = X_numeric[col].dropna().values.reshape(-1, 1)
            if len(col_data) > 0:
                self.scalers[col].partial_fit(col_data)
            
            current_mean = X_numeric[col].mean()
            if col in self.column_means:
                # Update running average
                n = len(X_numeric)
                total = self.column_means[col] * n + current_mean * n
                self.column_means[col] = total / (2 * n)
            else:
                self.column_means[col] = current_mean
        
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        
        X_numeric = X[self.numeric_columns].copy()
        
        for col in self.numeric_columns:
            col_data = X_numeric[col].values.reshape(-1, 1)
            nan_mask = np.isnan(col_data)
            col_data[nan_mask] = self.column_means[col]
            X_numeric[col] = self.scalers[col].transform(col_data).flatten()
            X_numeric.loc[nan_mask.flatten(), col] = np.nan
        
        X.loc[:,self.numeric_columns] = X_numeric
        return X

#load working scaler (joblib)
#proficiency_scaler = joblib.load('./opt/ml/model/proficiency_scaler.joblib')

if scalerUsed:
    with open('proficiency_scaler.json', 'r') as f:
        scaler_dict = json.load(f)

    proficiency_scaler = NanPreservingScaler()
    proficiency_scaler.scalers = {}
    for k, v in scaler_dict['scalers'].items():
        scaler = StandardScaler()
        scaler.mean_ = np.array(v['mean_'])
        scaler.scale_ = np.array(v['scale_'])
        scaler.var_ = np.array(v['var_'])
        proficiency_scaler.scalers[k] = scaler

    proficiency_scaler.column_means = scaler_dict['column_means']
    proficiency_scaler.numeric_columns = scaler_dict['numeric_columns']
else:
    proficiency_scaler=None


def apply_sigmoid(predictions, params):
    a, b = params
    return 1 / (1 + np.exp(-a * (predictions - b)))  # b is the centering point

if skillTransformerUsed:
    with open('sigmoid_params.json', 'r') as f:
        params = json.load(f)
        try:
            sigmoid_params = [params['a'], params['b']] 
        except:
            sigmoid_params = [params['a'], params['b'], params['future_window']]
            sigmoid_params = sigmoid_params[:1]
            
else:
    sigmoid_params = None

# Load the confidence score scaler
class PercentileRankCalculator:
    def __init__(self, scores):
        self.scores = np.array(scores)
        self.ranks = stats.rankdata(self.scores, method='average')
        self.total_scores = len(self.scores)
    
    def get_percentile_rank(self, new_scores):
        new_scores = np.atleast_1d(new_scores)
        
        # Find where the new scores would be inserted
        insert_positions = np.searchsorted(self.scores, new_scores)
        
        # Calculate ranks
        new_ranks = np.where(insert_positions == 0, 1,
                    np.where(insert_positions == self.total_scores, self.total_scores,
                    np.where(new_scores == self.scores[np.maximum(insert_positions - 1, 0)],
                             self.ranks[np.maximum(insert_positions - 1, 0)],
                             insert_positions + 1)))
        
        # Calculate percentile ranks
        percentile_ranks = (new_ranks / (self.total_scores + 1)) * 100
        
        return percentile_ranks

    #def save(self, filename):
    #    with open(filename, 'wb') as f:
    #        pickle.dump(self, f)
            
    def save(self, filename):
        data = {
            'scores': self.scores.tolist(),
            'ranks': self.ranks.tolist(),
            'total_scores': self.total_scores
        }
        with open(filename, 'w') as f:
            json.dump(data, f)            
    
#     @classmethod
#     def load(cls, filename):
#         with open(filename, 'rb') as f:
#             return pickle.load(f)
    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        obj = cls([])  # Create an empty instance
        obj.scores = np.array(data['scores'])
        obj.ranks = np.array(data['ranks'])
        obj.total_scores = data['total_scores']
        return obj    

    
# Load the confidence score scaler
with open('confidence_score_scaler.json', 'r') as f:
    scaler_data = json.load(f)
confidence_score_scaler = PercentileRankCalculator([])
confidence_score_scaler.scores = np.array(scaler_data['scores'])
confidence_score_scaler.ranks = np.array(scaler_data['ranks'])
confidence_score_scaler.total_scores = scaler_data['total_scores']    
    
    
# with open('./opt/ml/model/confidence_score_scaler.pkl', 'rb') as f:
#     confidence_score_scaler = pickle.load(f)
# confidence_score_scaler.save('./opt/ml/model/confidence_score_scaler.json')    



# Load the item parameters
item_params = pd.read_csv('item_params.csv')
#basically gets rid of 'version', other stored metadata(?)
item_params = item_params[['question_id', 'skill_id', 'discriminability', 'difficulty', 
                               'guessing','inattention', 'discriminability_error', 
                               'difficulty_error', 'guessing_error', 'inattention_error', 
                               'auc_roc', 'optimal_threshold', 'tpr', 'tnr', 'skill_optimal_threshold',
                               'student_mean_accuracy', 'sample_size']]

#create a copy where we pre-transform sample size (for horizontal batch inference)
transformed_loaded_ipd = item_params.copy()
transformed_loaded_ipd['LOG_sample_size'] = np.log(transformed_loaded_ipd['sample_size'])
transformed_loaded_ipd = transformed_loaded_ipd.drop('sample_size', axis=1)

# Make list of all valid skill_ids (for horiz batch inference)
valid_skill_ids = set(transformed_loaded_ipd['skill_id'].unique())
    
print("All models, artifacts, and IP data loaded successfully.")

def validate_input(data):
    """Validates and cleans input data."""
    required_fields = ['skillId', 'questionId', 'eventTime', 'questionIdsHistory', 
                      'correctnessHistory', 'durationSecondsHistory', 'eventTimesHistory']
    
    # Check for missing required fields
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        
    # Ensure all histories are lists
    for field in ['questionIdsHistory', 'correctnessHistory', 'durationSecondsHistory', 'eventTimesHistory']:
        if not isinstance(data[field], list):
            data[field] = [data[field]] if data[field] is not None else []

    return data
    
def process_input_original(data):
    """Process input data without lag shifting for skill predictions"""
    try:
        data = validate_input(data)
        
        features = {
            'SKILL': [data['skillId']],
            'QUESTIONID': [data['questionId']],
            'OCCURRED_AT': [pd.to_datetime(data['eventTime'])]
        }
        
        # Original lag structure (1 to HISTORY_LENGTH)
        for i in range(1, HISTORY_LENGTH + 1):
            features[f'QUESTIONID_LAG_{i}'] = [
                data['questionIdsHistory'][i-1] if i-1 < len(data['questionIdsHistory']) else np.nan
            ]
            features[f'CORRECTNESS_LAG_{i}'] = [
                float(data['correctnessHistory'][i-1]) 
                if i-1 < len(data['correctnessHistory']) and data['correctnessHistory'][i-1] is not None 
                else np.nan
            ]
            features[f'DURATIONSECONDS_LAG_{i}'] = [
                float(data['durationSecondsHistory'][i-1])
                if i-1 < len(data['durationSecondsHistory']) and data['durationSecondsHistory'][i-1] is not None
                else np.nan
            ]
            features[f'OCCURREDAT_LAG_{i}'] = [
                pd.to_datetime(data['eventTimesHistory'][i-1])
                if i-1 < len(data['eventTimesHistory']) and data['eventTimesHistory'][i-1]
                else pd.NaT
            ]
        
        return pd.DataFrame(features)
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

def process_input_lagged(data):
    """Process input data with lag shifting for item predictions"""
    #API is inexplicably, despite specifications 
    #https://illuminate.atlassian.net/wiki/spaces/~42953498/pages/17608311471/ELA+Student+Proficiency+Endpoint+Inputs+and+Outputs
    #double submitting the last item as history AND current
    #this will deliberately ignore the first history item ('lagged' model metrics)
    #but only for item, not the live skill prediction

    try:
        data = validate_input(data)
        
        features = {
            'SKILL': [np.nan if not data['skillId'] else data['skillId']],
            'QUESTIONID': [np.nan if not data['questionIdsHistory'] else data['questionIdsHistory'][0]],
            'OCCURRED_AT': [np.nan if not data['eventTimesHistory'] else pd.to_datetime(data['eventTimesHistory'][0])]
        }
        # Shifted lag structure (2 to HISTORY_LENGTH+1)
        # start w indx 1 instead of 0: [i-1], 2,HL+2: 1, HL+1
        for i in range(2, HISTORY_LENGTH + 2):
            features[f'QUESTIONID_LAG_{i-1}'] = [
                data['questionIdsHistory'][i-1] if i-1 < len(data['questionIdsHistory']) else np.nan
            ]
            features[f'CORRECTNESS_LAG_{i-1}'] = [
                float(data['correctnessHistory'][i-1]) 
                if i-1 < len(data['correctnessHistory']) and data['correctnessHistory'][i-1] is not None 
                else np.nan
            ]
            features[f'DURATIONSECONDS_LAG_{i-1}'] = [
                float(data['durationSecondsHistory'][i-1])
                if i-1 < len(data['durationSecondsHistory']) and data['durationSecondsHistory'][i-1] is not None
                else np.nan
            ]
            features[f'OCCURREDAT_LAG_{i-1}'] = [
                pd.to_datetime(data['eventTimesHistory'][i-1])
                if i-1 < len(data['eventTimesHistory']) and data['eventTimesHistory'][i-1]
                else pd.NaT
            ]
        
        return pd.DataFrame(features)
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")


def process_inference_data(df, item_params):
    # Load ItemParameters already loaded (item_params)
        
    # Step 0: CORRECTNESS
    correctness_columns = ['CORRECTNESS'] + [f'CORRECTNESS_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
    for col in correctness_columns:
        if col in df.columns:
            #df[col] = pd.to_numeric(df[col], errors='coerce') / 100
            #API is inexplicably, despite specifications 
            #https://illuminate.atlassian.net/wiki/spaces/~42953498/pages/17608311471/ELA+Student+Proficiency+Endpoint+Inputs+and+Outputs
            #dividiing by 100 before submission
            df[col] = pd.to_numeric(df[col], errors='coerce')
            #I have a lot of unit tests, test data, etc so this is a unoptimal patch for now for the API
            # Normalize only if the column is not empty and max > 1
            if len(df[col].dropna()) > 0 and df[col].max() > 1:
                df[col] = df[col] / 100

    # Step 1: Calculate time differences
    occurred_columns = ['OCCURRED_AT'] + [f'OCCURREDAT_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]

    #timezone formatting issues
    for col in occurred_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            if df[col].dt.tz is None:
                df[col] = pd.DatetimeIndex(df[col]).tz_localize('UTC')
            else:
                df[col] = df[col].dt.tz_convert('UTC')
   
    for i in range(1, len(occurred_columns)):
        if occurred_columns[i-1] in df.columns and occurred_columns[i] in df.columns:
            diff_col = f'OCCURREDAT_DIFF_{i}'
            #df[diff_col] = (pd.to_datetime(df[occurred_columns[i-1]]) - pd.to_datetime(df[occurred_columns[i]])).dt.total_seconds() / 3600
            #df[diff_col] = ((df[occurred_columns[i-1]] - df[occurred_columns[i]]).dt.total_seconds() / 3600).round(6)
            # Make the operations more explicit
            time_diff_seconds = (df[occurred_columns[i-1]] - df[occurred_columns[i]]).dt.total_seconds()
            time_diff_hours = time_diff_seconds / 3600
            df[diff_col] = np.round(time_diff_hours, decimals=3)
        
    # Drop OCCURRED_AT columns
    df = df.drop(columns=[col for col in occurred_columns if col in df.columns])

    # Step 3a: Minimum cap OCCURREDAT_DIFF to 1/3600
    diff_columns = [col for col in df.columns if col.startswith('OCCURREDAT_DIFF')]
    for col in diff_columns:
        df[col] = df[col].clip(lower=1/3600)

    # Step 3b and 3c: Cap DURATIONSECONDS
    duration_columns = [f'DURATIONSECONDS_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
    for col in duration_columns:
        if col in df.columns:
            df[col] = df[col].clip(lower=1, upper=300)

    # Step 4: Log transform
    for col in diff_columns + duration_columns:
        if col in df.columns:
            df[f'LOG_{col}'] = np.log(df[col])

    # Step 5: LEFT JOIN with ItemParameters
    for i in range(HISTORY_LENGTH+1): #0 is current, 1-10 lag 
        suffix = f'_LAG_{i}' if i > 0 else ''
        question_col = 'QUESTIONID' if i == 0 else f'QUESTIONID_LAG_{i}'
        if question_col in df.columns:
            df[question_col] = df[question_col].fillna('nan').astype(str)
            temp_df = item_params.copy()
            temp_df.columns = [f'{col}{suffix}' for col in temp_df.columns]
            df = df.merge(temp_df, left_on=[question_col, 'SKILL'], 
                          right_on=[f'question_id{suffix}', f'skill_id{suffix}'], how='left')
            
            # Drop the join columns
            df = df.drop(columns=[f'question_id{suffix}', f'skill_id{suffix}'])
            
    # Step 5b: EXTRA STEP FOR INFERENCE: DROP questions
    qid_columns = [f'QUESTIONID_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
    df = df.drop(columns=[col for col in qid_columns if col in df.columns])

    # Step 6: Log transform sample_size and remove original
    sample_size_columns = ['sample_size'] + [f'sample_size_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
    for col in sample_size_columns:
        if col in df.columns:
            df[f'LOG_{col}'] = np.log(df[col])
            df = df.drop(columns=[col])

    # Step 7: Add spread features
    for feature in ['difficulty', 'optimal_threshold', 'student_mean_accuracy', 'tnr', 'tpr']:
        cols = [f'{feature}_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
        if all(col in df.columns for col in cols):
            df[f'{feature}_spread'] = df[cols].max(axis=1) - df[cols].min(axis=1)

    # Step 8: Add mean of auc_roc and discriminability and error
    auc_cols = [f'auc_roc_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
    disc_cols = [f'discriminability_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
    if all(col in df.columns for col in auc_cols):
        df['mean_auc_roc'] = df[auc_cols].mean(axis=1)
    if all(col in df.columns for col in disc_cols):
        df['mean_discriminability'] = df[disc_cols].mean(axis=1)

    error_features = ['discriminability_error', 'difficulty_error', 'guessing_error', 'inattention_error']
    for feature in error_features:
        feature_cols = [f'{feature}_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
        if all(col in df.columns for col in feature_cols):
            df[f'mean_{feature}'] = df[feature_cols].mean(axis=1)

    # Step 9: drop redundant skill OTs
    skill_optimal_threshold_columns = [f'skill_optimal_threshold_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
    df = df.drop(columns=[col for col in skill_optimal_threshold_columns if col in df.columns])

    # Step 10: add num responses
    correctness_columns = [f'CORRECTNESS_LAG_{i}' for i in range(1, HISTORY_LENGTH+1)]
    df['num_responses'] = df[[col for col in correctness_columns if col in df.columns]].notna().sum(axis=1).astype(float)

    return df
    

def run_model_inference(df, proficiency_model, confidence_model, proficiency_scaler, confidence_score_scaler):
    """
    Run model inference on a single item/skill combination.
    """
    
    if scalerUsed:
        # Scale data with proficiency_scaler
        scaled_data = proficiency_scaler.transform(df)
    else:
        scaled_data = df[:]
    
    # Run inference using proficiency_model
    dmatrix = xgb.DMatrix(scaled_data)
    if skillTransformerUsed:
        item_prediction = apply_sigmoid(proficiency_model.predict(dmatrix),sigmoid_params)
    else:
        item_prediction = np.clip(proficiency_model.predict(dmatrix),0,1)
    
    # Run inference using confidence_model
    confidence_input = scaled_data.copy()
    confidence_input['proficiency_model_prediction'] = item_prediction
    confidence_dmatrix = xgb.DMatrix(confidence_input)
    item_prediction_error = np.clip(confidence_model.predict(confidence_dmatrix),0,1)
    
    # Scale the confidence score
    item_prediction_confidence = confidence_score_scaler.get_percentile_rank(1- item_prediction_error)

    return item_prediction, item_prediction_error, item_prediction_confidence

def batch_predict_all_questions_for_skills_w_conf(proficiency_model, confidence_model, 
                                                  transformed_loaded_ipd, X_input, skill_ids, 
                                                  feature_names, proficiency_scaler):
    unique_skill_ids = np.unique(skill_ids)
    all_predictions = []
    all_question_ids = []
    
    #we use transformed loaded ipd (which just pre-log txs sample size)
    #and ignores version
    transformed_ipd_features = ['discriminability', 'difficulty', 'guessing', 'inattention',
                    'discriminability_error', 'difficulty_error', 'guessing_error',
                    'inattention_error', 'auc_roc', 'optimal_threshold', 'tpr', 'tnr',
                    'student_mean_accuracy', 'LOG_sample_size']
    
    
        
    # Pre-process IPD data/condense to what we need
    transformed_ipd_dict = {skill: transformed_loaded_ipd[transformed_loaded_ipd['skill_id'] == skill].groupby('question_id').last() for skill in unique_skill_ids}
    
    prediction_counts = []  # To keep track of number of predictions per input row
    valid_indices = [] # to track bad submissions with no valid skill_id
    
    for i, skill_id in enumerate(skill_ids):
        if skill_id not in valid_skill_ids:
            prediction_counts.append(0)
            continue

        valid_indices.append(i)
        X_skill = X_input.iloc[i:i+1]  # Get single row
        
        question_ids = transformed_ipd_dict[skill_id].index
        n_questions = len(question_ids)
        
        if n_questions == 0:
            prediction_counts.append(0)
            continue
            
        # Create an array to hold all question variations
        all_questions = np.repeat(X_skill.values, n_questions, axis=0)
        
        # Update the question-specific columns for each question
        for col in transformed_ipd_features:
            all_questions[:, feature_names.index(col)] = transformed_ipd_dict[skill_id][col].values
        
        all_predictions.append(all_questions)
        all_question_ids.extend(question_ids)
        prediction_counts.append(n_questions)
    
    #possibly unnecessary check -- return empties if function got called w no valid skill_id
    if not all_predictions:
        return (np.array([]), np.array([]), np.array([]), [], prediction_counts, valid_indices)

    # Combine all predictions
    all_predictions = np.vstack(all_predictions)
    
    #convert to df
    all_predictions = pd.DataFrame(all_predictions, columns=feature_names)
    
    if scalerUsed:
        # Scale the data
        X_all_questions_scaled = proficiency_scaler.transform(all_predictions)
    else:
        X_all_questions_scaled = all_predictions.copy()
        

    # Make predictions for all questions at once
    dmatrix = xgb.DMatrix(X_all_questions_scaled)
    
    #clip or sigmoid the predictions
    if skillTransformerUsed:
        predictions = apply_sigmoid(proficiency_model.predict(dmatrix),sigmoid_params)
    else:
        predictions = np.clip(proficiency_model.predict(dmatrix),0,1)

    
    # Prepare input for confidence model
    #confidence_input = np.column_stack((X_all_questions_scaled, predictions))
    X_all_questions_scaled['proficiency_model_prediction']=predictions
    
    # Make confidence predictions
    confidence_dmatrix = xgb.DMatrix(X_all_questions_scaled)
    confidence_predicted_error = np.clip(confidence_model.predict(confidence_dmatrix),0,1)
    
    confidence_scores = confidence_score_scaler.get_percentile_rank(1 - confidence_predicted_error)

    
    return predictions, confidence_predicted_error, confidence_scores, all_question_ids, prediction_counts, valid_indices

def run_inference(data):
    """
    Run comprehensive inference on input data.
    """
    # Process the input data two ways
    if isinstance(data, dict):  # Single inference
        input_df_original = process_input_original(data)
        input_df_lagged = process_input_lagged(data)
    elif isinstance(data, list):  # Batch inference
        input_df_original = pd.concat([process_input_original(item) for item in data], ignore_index=True)
        input_df_lagged = pd.concat([process_input_lagged(item) for item in data], ignore_index=True)
    else:
        raise ValueError("Input data must be a dictionary for single inference or a list of dictionaries for batch inference")
            
    # Run lagged item inference (for metrics tracking)
    # Process inference data
    processed_df_lagged = process_inference_data(input_df_lagged, item_params)    
    processed_df_lagged = processed_df_lagged.drop(['QUESTIONID', 'SKILL'], axis=1)
    # Reorder features
    correct_feature_order = proficiency_model.feature_names
    reordered_df_lagged = processed_df_lagged[correct_feature_order]
    
    
    lagged_item_predictions, lagged_item_prediction_errors, lagged_item_prediction_confidences = run_model_inference(
        reordered_df_lagged, proficiency_model, confidence_model, proficiency_scaler, confidence_score_scaler
    )
        
    # Run batch prediction for skills (using original)
    if batchSkillUsed:
        
        # Process inference data 
        processed_df_original = process_inference_data(input_df_original, item_params)
        # Extract required information for batch prediction (using original)
        skill_ids = processed_df_original['SKILL'].tolist()
        processed_df_original = processed_df_original.drop(['QUESTIONID', 'SKILL'], axis=1)
        # Reorder features 
        reordered_df_original = processed_df_original[correct_feature_order]
        feature_names = reordered_df_original.columns.tolist()

        
        predictions, confidence_predicted_error, confidence_scores, all_question_ids, prediction_counts, valid_indices = batch_predict_all_questions_for_skills_w_conf(
            proficiency_model, confidence_model, transformed_loaded_ipd, reordered_df_original, 
            skill_ids, feature_names, proficiency_scaler
        )

        # Process skill predictions as before
        skill_predictions_means = np.full(len(skill_ids), np.nan)
        skill_uncertainity_means = np.full(len(skill_ids), np.nan)
        skill_confidence_score_means = np.full(len(skill_ids), np.nan)

        if len(predictions) > 0:
            valid_predictions = np.split(predictions, np.cumsum(prediction_counts)[:-1])
            valid_uncertainties = np.split(confidence_predicted_error, np.cumsum(prediction_counts)[:-1])
            valid_confidences = np.split(confidence_scores, np.cumsum(prediction_counts)[:-1])

            for i, idx in enumerate(valid_indices):
                if prediction_counts[idx] > 0:
                    skill_predictions_means[idx] = np.mean(valid_predictions[i])
                    skill_uncertainity_means[idx] = np.mean(valid_uncertainties[i])
                    skill_confidence_score_means[idx] = np.mean(valid_confidences[i])
    #else just do 'item agnostic' prediction
    else:        
        # Process inference data WITHOUT next-item information (API IS INCORRECT HERE TOO)
        #DROP Erroneous questionid first, before processing
        processed_df_original = input_df_original.drop(['QUESTIONID'], axis=1)
        processed_df_original = process_inference_data(input_df_original, item_params)
        processed_df_original = processed_df_original.drop(['SKILL'], axis=1)
        # Extract required information for batch prediction (using original)
        # Reorder features 
        reordered_df_original = processed_df_original[correct_feature_order]
        feature_names = reordered_df_original.columns.tolist()


        skill_predictions_means, skill_uncertainity_means, skill_confidence_score_means = run_model_inference(
            reordered_df_original, proficiency_model, confidence_model, proficiency_scaler, confidence_score_scaler
        )


    output_df = pd.DataFrame({
        'item_prediction': lagged_item_predictions,
        'item_prediction_error': lagged_item_prediction_errors,
        'item_prediction_confidence': lagged_item_prediction_confidences,
        'skill_prediction': skill_predictions_means,
        'skill_prediction_error': skill_uncertainity_means,
        'skill_prediction_confidence': skill_confidence_score_means
    })
    
    output_df = output_df.replace({np.nan: None})
    
    if debugInputs:
        return output_df, reordered_df_lagged
    else:
        return output_df
#
#Inference Wrapper ends here
#
@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy.
    """
    return flask.Response(response='\n', status=200, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data.
    """
    debug_info = {
        'received_content_type': flask.request.content_type,
        'received_data_length': len(flask.request.get_data()),
        'received_raw_data': flask.request.get_data().decode('utf-8', errors='replace')
    }
    
    if flask.request.content_type != 'application/json':
        error_response = {
            'error': 'This predictor only supports JSON data',
            'debug_info': debug_info
        }
        return flask.Response(
            response=json.dumps(error_response),
            status=415,
            mimetype='application/json'
        )
    
    try:
        data = flask.request.get_json()
        debug_info['parsed_json_length'] = len(str(data))
        debug_info['parsed_json'] = data
        
        if debugInputs: #debug
            results, feature_engineered_input = run_inference(data)
            #force only one row 
            #debug_info['feature_engineered_input'] = feature_engineered_input.iloc[0].to_dict()
            debug_info['feature_engineered_input'] = feature_engineered_input.iloc[0].replace({np.nan: None}).to_dict()
        else:
            results = run_inference(data)
        
        results['model_version'] = model_version
        
        # Add debug info to successful response while keeping original structure
        response_data = {
            'prediction': results.to_dict(orient='list'),
            'debug_info': debug_info
        }
        
        return jsonify(response_data)
        
    except json.JSONDecodeError as e:
        error_response = {
            'error': 'Failed to parse JSON data',
            'error_details': {
                'error_type': 'JSONDecodeError',
                'error_message': str(e),
                'error_position': f'line {e.lineno}, column {e.colno}',
                'error_doc': e.doc
            },
            'debug_info': debug_info
        }
        return flask.Response(
            response=json.dumps(error_response),
            status=400,
            mimetype='application/json'
        )
    except Exception as e:
        error_response = {
            'error': 'An unexpected error occurred while processing the request',
            'error_details': {
                'error_type': type(e).__name__,
                'error_message': str(e)
            },
            'debug_info': debug_info
        }
        return flask.Response(
            response=json.dumps(error_response),
            status=500,
            mimetype='application/json'
        )

@app.errorhandler(404)
def not_found_error(error):
    return flask.Response(
        response=json.dumps({
            'error': 'Resource not found',
            'error_details': {
                'error_type': '404',
                'error_message': 'The requested URL was not found on the server.'
            }
        }),
        status=404,
        mimetype='application/json'
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
