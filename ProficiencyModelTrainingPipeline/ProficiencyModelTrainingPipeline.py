import sys
import subprocess

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import xgboost as xgb
except:
    install_package("xgboost")

try:
    import os, sys
    #sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SnowflakeETL'))

    import SnowflakeETL
except:
    install_package("snowflake")
    import SnowflakeETL
    

import pandas as pd
import numpy as np
import time
import os
import glob
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import glob
import os
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
import glob
import json
import logging
from datetime import datetime
from scipy import stats
import pickle
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count
from functools import partial


import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

#test train split
#test train split (pre feature engineering, pre irtsdt generation)


import pyarrow.parquet as pq

def process_file_for_studentids(file, column_names):
    try:
        # Read Parquet file metadata
        parquet_file = pq.ParquetFile(file)
        file_schema = parquet_file.schema.names
        
        # Find the index of 'STUDENTID' in the provided column names
        studentid_index = column_names.index('STUDENTID')
        
        # Map the 'STUDENTID' to the actual column name in the file
        actual_studentid_column = file_schema[studentid_index]
        
        # Read only the student ID column
        df = pd.read_parquet(file, columns=[actual_studentid_column])
        
        # Rename the column to 'STUDENTID'
        df.columns = ['STUDENTID']
        
        return df['STUDENTID'].unique()
    except Exception as e:
        logger.info(f"Error processing file {file}: {str(e)}")
        return np.array([])

def read_unique_studentids_from_parquet_files(file_pattern, column_names, batch_size=100):
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        logger.info(f"No files found matching the pattern: {file_pattern}")
        raise ValueError(f"No files found matching the pattern: {file_pattern}")
    
    # Create a partial function with pre-filled arguments
    process_file_partial = partial(process_file_for_studentids, column_names=column_names)
    
    # Use all available CPU cores minus 2, but not less than 1
    num_cores = max(1, cpu_count() - 3)
    
    unique_studentids = set()
    
    # Process files in batches
    total_files = len(all_files)
    for i in range(0, total_files, batch_size):
        batch_files = all_files[i:i+batch_size]
        
        with Pool(num_cores) as pool:
            results = pool.map(process_file_partial, batch_files)
        
        for studentids in results:
            unique_studentids.update(studentids)
        
        logger.info(f"Processed {min(i+batch_size, total_files)} out of {total_files} files. Current unique IDs: {len(unique_studentids)}")
    
    return pd.Series(list(unique_studentids), name='STUDENTID')



#IRT-SDT Calculations
def process_file_for_skills(file, column_names, train_ids):
    try:
        parquet_file = pq.ParquetFile(file)
        file_schema = parquet_file.schema.names
        
        studentid_col = file_schema[column_names.index('STUDENTID')]
        skill_col = file_schema[column_names.index('SKILL')]
        
        df = pd.read_parquet(file, columns=[studentid_col, skill_col])
        df.columns = ['STUDENTID', 'SKILL']
        
        df = df[df['STUDENTID'].isin(train_ids)]
        
        return set(df['SKILL'])
    except Exception as e:
        logger.info(f"Error processing file {file} for skills: {str(e)}")
        return set()

def get_distinct_skills_from_parquets(file_pattern, column_names, train_ids, batch_size=100):
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        logger.info(f"No files found matching the pattern: {file_pattern}")
        raise ValueError(f"No files found matching the pattern: {file_pattern}")
    
    process_file_partial = partial(process_file_for_skills, column_names=column_names, train_ids=train_ids)
    
    num_cores = max(1, cpu_count() - 3)
    
    logger.info(f'Starting {num_cores} cores for skill search.')
    unique_skills = set()
    
    total_files = len(all_files)
    for i in range(0, total_files, batch_size):
        batch_files = all_files[i:i+batch_size]
        
        with Pool(num_cores) as pool:
            batch_results = pool.map(process_file_partial, batch_files)
        
        for skills in batch_results:
            unique_skills.update(skills)
        
        logger.info(f"Processed {min(i+batch_size, total_files)} out of {total_files} files. Current unique skills: {len(unique_skills)}")
    
    return list(unique_skills)

def process_file_for_skill_data(file, skill, column_names, train_ids):
    try:
        parquet_file = pq.ParquetFile(file)
        file_schema = parquet_file.schema.names
        
        studentid_col = file_schema[column_names.index('STUDENTID')]
        questionid_col = file_schema[column_names.index('QUESTIONID')]
        correctness_col = file_schema[column_names.index('CORRECTNESS')]
        skill_col = file_schema[column_names.index('SKILL')]
        
        df = pd.read_parquet(file, columns=[studentid_col, questionid_col, correctness_col, skill_col])
        df.columns = ['STUDENTID', 'QUESTIONID', 'CORRECTNESS', 'SKILL']
        
        skill_df = df[df['SKILL'] == skill]
        skill_df = skill_df[skill_df['STUDENTID'].isin(train_ids)]
        
        if not skill_df.empty:
            result = skill_df[['STUDENTID', 'QUESTIONID', 'CORRECTNESS']].copy()
            result['CORRECTNESS'] = pd.to_numeric(result['CORRECTNESS'], errors='coerce')
            return result
        return None
    except Exception as e:
        logger.info(f"Error processing file {file} for skill data: {str(e)}")
        return None

def get_data_for_skill(file_pattern, skill, column_names, train_ids, DATA_LIMIT=None, batch_size=100):
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        logger.info(f"No files found matching the pattern: {file_pattern}")
        raise ValueError(f"No files found matching the pattern: {file_pattern}")
    
    process_file_partial = partial(process_file_for_skill_data, skill=skill, column_names=column_names, train_ids=train_ids)
    
    num_cores = max(1, cpu_count() - 3)
    
    result_dfs = []
    total_rows = 0
    total_files = len(all_files)
    
    for i in range(0, total_files, batch_size):
        batch_files = all_files[i:i+batch_size]
        
        with Pool(num_cores) as pool:
            batch_results = pool.map(process_file_partial, batch_files)
        
        for df in batch_results:
            if df is not None and not df.empty:
                result_dfs.append(df)
                total_rows += len(df)
                
                if DATA_LIMIT is not None and total_rows >= DATA_LIMIT:
                    break
        
        logger.info(f"Processed {min(i+batch_size, total_files)} out of {total_files} files. Current rows: {total_rows}")
        
        if DATA_LIMIT is not None and total_rows >= DATA_LIMIT:
            break
    
    if not result_dfs:
        return pd.DataFrame(columns=['STUDENTID', 'QUESTIONID', 'CORRECTNESS'])
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    if DATA_LIMIT is not None:
        result = result.head(DATA_LIMIT)
    
    logger.info(f'Final shape for skill {skill}: {np.shape(result)}')
    
    return result


def process_skills_irtsdt(file_pattern, column_names, output_filename, version, train_ids, 
                       binary=False, restart_index=0, TestMode=False, DATA_LIMIT=None):

    # Set up logging
    import os, sys
    log_file = os.path.join('irtsdtlog_'+str(datetime.now().strftime("%Y%m%d_%H%M%S"))+'.txt')
    setup_logger(log_file)

    # Set up logging
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Import custom IRT library (adjust the path as needed)
    try:
        import os, sys
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ItemParametersCalculate'))

        #current_dir = Path.cwd()
        #base_path = os.path.join(os.path.dirname(current_dir), '../..')
        #irt_dir = os.path.join(os.path.dirname(current_dir), 'ItemParametersCalculate')
        #sys.path.append(str(irt_dir))
        import customPyIRT
    except:
        try:
            current_dir = os.getcwd()
            base_path = os.path.abspath(os.path.join(current_dir, '..'))
            sys.path.append(os.path.join(base_path, 'ItemParametersCalculate'))
            import customPyIRT
        except:
            print('IRT library not found.')


    logging.info(f"Starting IRT processing. Output: {output_filename}, Version: {version}")
    logging.info(f"Number of training IDs: {len(train_ids)}")

    # Get all unique skills
    all_skills = get_distinct_skills_from_parquets(file_pattern, column_names, train_ids)
    logging.info(f"{len(all_skills)} skills found.")
    
    #check if file already calculated
    if os.path.exists(output_filename):
        logging.info("IRT file already exists...looking for last skill_id run.")
        irt_df = pd.read_csv(output_filename)
        # Check if the DataFrame is empty
        if irt_df.empty:
            logging.info("The CSV file is empty.")
            restart_index=0
        else:
            # Get the last row
            #last_row = irt_df.iloc[-1]
            # Extract the skill_id from the last row
            #last_skill_id = last_row['skill_id']

            calculated_skill_ids = set(irt_df['skill_id'])
            logging.info(f"There are {len(calculated_skill_ids)} skills in existing IP file.")
            restart_index=0

            all_skills = set(all_skills)
            all_skills = all_skills - calculated_skill_ids
            all_skills = list(all_skills)
            logging.info(f"There are {len(all_skills)} skills missing to be computed.")

            #index = all_skills.index(last_skill_id) if last_skill_id in all_skills else -1
            #start on next skill_id not calc yet
            #restart_index=index+1

    
    #see if file exists, grab all computed skills
    try:
        with open(output_filename) as f:
            logging.info(f"Existing file found: {output_filename}.")            
            computed_skills = set(line.split(',')[1] for line in f.readlines()[1:])
    except FileNotFoundError:
        computed_skills = set()
        logging.info(f"No calculated skills found. Will write parameters to: {output_filename}.")            
        
    
    if TestMode:
        logging.info(f"Test Mode. Runnning 2 skills.")
        all_skills=all_skills[:2]
        

    # Prepare for IRT processing
    skills_calculated = 0
    count = restart_index

    # Process each skill
    for skill_id in all_skills[restart_index:]:
        logging.info(f'Processing skill #{count+1}: {skill_id}')
        
        if skill_id in computed_skills:
            logging.info(f'Existing computation found for skill #{count+1}: {skill_id}. Skipping.')
        else:
            
            try:
                # Get data for this skill
                start = time.time()
                df = get_data_for_skill(file_pattern, skill_id, column_names, train_ids, DATA_LIMIT=DATA_LIMIT)

                # Filter for train_ids
                #df = df[df['STUDENTID'].isin(train_ids)]

                logging.info(f'Loading time: {time.time()-start:.2f} seconds for {len(df)} rows after filtering')

                # Prepare data for IRT
                irt_df = df.rename(columns={
                    'STUDENTID': 'student_id',
                    'QUESTIONID': 'math_question_id',
                    'CORRECTNESS': 'correctness'
                })
                irt_df['skill_id'] = skill_id

                # Create IRT table
                if binary:
                    table = customPyIRT.returnTable(irt_df)
                else:
                    table = customPyIRT.returnTable(irt_df, False)
                logging.info(f'Table size: {np.shape(table)}')

                # Solve IRT
                logging.info('Solving for item parameters...')
                starttime = time.time()
                solvedIRT_7 = customPyIRT.solve_IRT_for_matrix(
                    table, all_thetas=None, iterations=250, 
                    FOUR_PL=True, show_convergence=0,
                    bounds=((1,-3,0,.5),(100,3,.5,1))
                )
                logging.info(f'Parameter calculation time: {time.time()-starttime:.2f} seconds')

                # Check for NaNs
                nan_checks = [
                    ('est params', solvedIRT_7.est_params[1]),
                    ('auc_roc', solvedIRT_7.auc_roc),
                    ('optimal_threshold', solvedIRT_7.optimal_threshold)
                ]
                for name, arr in nan_checks:
                    nan_count = np.isnan(arr).sum()
                    if nan_count > 0:
                        logging.warning(f'NaNs in {name}: {nan_count} out of {np.size(arr)}')

                # Export results
                customPyIRT.export_object_to_csv(solvedIRT_7, skill_id, output_filename, version)
                skills_calculated += 1

            except Exception as e:
                logging.error(f'Error processing skill {skill_id}: {str(e)}')

        count += 1

    logging.info(f'Finished processing {count} skills. Successfully calculated parameters for {skills_calculated} skills.')
    logging.info('IRT processing completed.')



#
# Feature engineering
def process_parquet_files(input_pattern, output_folder, train_ids, item_params_file, 
                          column_names, TestMode, FUTURE_WINDOW):
    # Load ItemParameters
    item_params = pd.read_csv(item_params_file)
    item_params = item_params[['question_id', 'skill_id', 'discriminability', 'difficulty', 'guessing', 'inattention', 
                               'discriminability_error', 'difficulty_error', 'guessing_error', 'inattention_error', 
                               'auc_roc', 'optimal_threshold', 'tpr', 'tnr', 'skill_optimal_threshold', 
                               'student_mean_accuracy', 'sample_size']]

    firstRun=True
    files_processed=0
    all_files = glob.glob(input_pattern)
#     if TestMode:
#         all_files = all_files_actual[:2]
#         print('test mode running...only processing a small set')
#     else:
#         all_files = all_files_actual       
        
    for i, file in enumerate(all_files):
    #for file in all_files:
        df = pd.read_parquet(file)
        df.columns = column_names

        # Filter by STUDENTID inclusion in train_ids
        df = df[df['STUDENTID'].isin(train_ids)]

        #Step 0: CORRECTNESS
        correctness_columns = ['CORRECTNESS'] + [f'CORRECTNESS_LAG_{i}' for i in range(1, 11)]
        for col in correctness_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100
            
        #Step 0b: do the same for future data
        # First check if any future correctness columns exist in the dataframe
        future_correctness_columns = [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]
        existing_future_cols = [col for col in future_correctness_columns if col in df.columns]
        if existing_future_cols:
            for col in existing_future_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100

        # Step 1: Calculate time differences
        occurred_columns = ['OCCURREDAT'] + [f'OCCURREDAT_LAG_{i}' for i in range(1, 11)]
        
        #timezone formatting matching to live inference requirements
        for col in occurred_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                if df[col].dt.tz is None:
                    df[col] = pd.DatetimeIndex(df[col]).tz_localize('UTC')
                else:
                    df[col] = df[col].dt.tz_convert('UTC')
                
        for i in range(1, len(occurred_columns)):
            diff_col = f'OCCURREDAT_DIFF_{i}'
            #df[diff_col] = ((df[occurred_columns[i-1]] - df[occurred_columns[i]]).dt.total_seconds() / 3600).round(6)
            # Make the operations more explicit
            time_diff_seconds = (df[occurred_columns[i-1]] - df[occurred_columns[i]]).dt.total_seconds()
            time_diff_hours = time_diff_seconds / 3600
            df[diff_col] = np.round(time_diff_hours, decimals=3)
        
        # Drop OCCURREDAT
        df = df.drop(columns=occurred_columns)

        # Step 3a: Minimum cap OCCURREDAT_DIFF to 1/3600
        diff_columns = [col for col in df.columns if col.startswith('OCCURREDAT_DIFF')]
        for col in diff_columns:
            df[col] = df[col].clip(lower=1/3600)
        
        # Step 3b and 3c: Cap DURATIONSECONDS
        #drop current DurationSeconds (future information)
        df = df.drop(columns=['DURATIONSECONDS'])
        duration_columns = [f'DURATIONSECONDS_LAG_{i}' for i in range(1, 11)]
        for col in duration_columns:
            df[col] = df[col].clip(lower=1, upper=300)
        
        # Step 4: Log transform
        for col in diff_columns + duration_columns:
            df[f'LOG_{col}'] = np.log(df[col])
        
        # Step 5: LEFT JOIN with ItemParameters
        for i in range(11):
            suffix = f'_LAG_{i}' if i > 0 else ''
            temp_df = item_params.copy()
            temp_df.columns = [f'{col}{suffix}' for col in temp_df.columns]
            df = df.merge(temp_df, left_on=[f'QUESTIONID{suffix}', f'SKILL'], 
                          right_on=[f'question_id{suffix}', f'skill_id{suffix}'], how='left')
            
         # Drop the join columns
            df = df.drop(columns=[f'question_id{suffix}', f'skill_id{suffix}'])

        # Step 6: Log transform sample_size and remove original
        sample_size_columns = ['sample_size'] + [f'sample_size_LAG_{i}' for i in range(1, 11)]
        for col in sample_size_columns:
            df[f'LOG_{col}'] = np.log(df[col])
            df = df.drop(columns=[col])
        
        # Step 7: Add spread features
        for feature in ['difficulty', 'optimal_threshold', 'student_mean_accuracy', 'tnr', 'tpr']:
            cols = [f'{feature}_LAG_{i}' for i in range(1, 11)]
            df[f'{feature}_spread'] = df[cols].max(axis=1) - df[cols].min(axis=1)
            #df[f'{feature}_spread'] = df[cols].apply(lambda row: np.nanmax(row) - np.nanmin(row) if not row.isna().all() else np.nan, axis=1)
            
        # Step 8: Add mean of auc_roc and discriminability and error
        auc_cols = [f'auc_roc_LAG_{i}' for i in range(1, 11)]
        disc_cols = [f'discriminability_LAG_{i}' for i in range(1, 11)]
        df['mean_auc_roc'] = df[auc_cols].mean(axis=1)
        df['mean_discriminability'] = df[disc_cols].mean(axis=1)

        error_features = ['discriminability_error', 'difficulty_error', 'guessing_error', 'inattention_error']
        for feature in error_features:
            feature_cols = [f'{feature}_LAG_{i}' for i in range(1, 11)]
            df[f'mean_{feature}'] = df[feature_cols].mean(axis=1)
            
        # Step 9: drop redundant skill OTs
        skill_optimal_threshold_columns = [f'skill_optimal_threshold_LAG_{i}' for i in range(1, 11)]
        for col in skill_optimal_threshold_columns:
            df = df.drop(columns=[col])
            
        #Step 10: add num responses
        correctness_columns = [f'CORRECTNESS_LAG_{i}' for i in range(1, 11)]
        df['num_responses'] = df[correctness_columns].notna().sum(axis=1)
        
        # Write updated parquet to new folder
        # Skip empty files after filtering
        if df.empty:
            logger.info(f'No matching IDs found in {file}, skipping...')
        elif len(df)<10:
            logger.info(f'File {file} has only {len(df)} rows, skipping...')
        else:
            output_file = os.path.join(output_folder, os.path.basename(file))
            df.to_parquet(output_file, index=False)
            logger.info(f'finished file: {file}')
            files_processed+=1
        
        # Save column names
        import json
        if firstRun:
            column_names_exported = df.columns.tolist()
            with open(os.path.join(output_folder, 'feature_engineered_column_names.json'), 'w') as f:
                json.dump(column_names_exported, f)
            firstRun=False
            
        if TestMode and (files_processed>=2):
            logger.info(f'test mode running...processed {files_processed} files. exiting feature processing...')
            break


#Scaler and Proficiency Model fit 


    
class LoggingCallback(xgb.callback.TrainingCallback):
    def __init__(self, logger, metric='rmse', log_interval=10):
        self.logger = logger
        self.metric = metric
        self.log_interval = log_interval
        self.metric_name = 'RMSE' if metric == 'rmse' else 'LogLoss'




#apparnetly no CE function??? in python
def cross_entropy(actual, predicted):
    import numpy as np
    epsilon = 1e-7  # Small epsilon value to avoid log(0)

    predicted = np.clip(predicted, epsilon, 1 - epsilon)  # Clip predicted values to avoid extreme values
    errors = actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)

    return -1*np.mean(errors)

def reg_custom_metric(preds, dtrain, triggerFull=False, DirectCall=False, tgt_threshold=0.5, pred_threshold=0.5, label=None):
    #triggerFull: directly call this func and calculate everything
    #label = set this directly to target values to use this function & ignore dmatrix
    
    if label is None:
        labels = dtrain.get_label()
    else:
        labels=label[:]
            
    #filter both based on nan predictions
    valid_mask = ~np.isnan(preds)
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    # Always calculate the primary metric
    # Regression metrics
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)    

    if not hasattr(reg_custom_metric, 'count'):
        reg_custom_metric.count = 0
    else:  
        reg_custom_metric.count += 1

    # Determine if this is a train or test call
    #is_train = (reg_custom_metric.count % 2 == 0)
    #iteration = reg_custom_metric.count // 2

    # Log the primary metric for every iteration
    #set_type = "Train" if is_train else "Test"

    # Log all metrics every 10 iterations
    #if ((iteration % 10 == 0) or (triggerFull)):
    if True:
        raw_score = preds
        probabilities = np.clip(preds, 0, 1)  # Ensure values are between 0 and 1
        classifications = (probabilities > pred_threshold).astype(int)
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        log_odds = np.log(probabilities) - np.log(1 - probabilities)
        #log_odds = np.log(probabilities / (1 - probabilities))  # Add small epsilon to avoid division by zero
        #log_odds = np.log(probabilities / (1 - probabilities + 1e-7))  # Add small epsilon to avoid division by zero
        #for imputed stats, we'll just binarize labels, since we've normalized target
        label_classifications = (labels > tgt_threshold).astype(int)

        mse = mean_squared_error(labels, raw_score)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, raw_score)
        r2 = r2_score(labels, raw_score)
        crossentropy = cross_entropy(labels, probabilities)
        
        try:
            logloss = log_loss(label_classifications, probabilities)
            accuracy = accuracy_score(label_classifications, classifications)
    #         precision = precision_score(label_classifications, classifications, average='binary',zero_division=0)
    #         recall = recall_score(label_classifications, classifications, average='binary')
            # Get precision/recall for both classes separately
            precision_by_class = precision_score(label_classifications, classifications, average=None, zero_division=0)
            recall_by_class = recall_score(label_classifications, classifications, average=None)

            f1 = f1_score(label_classifications, classifications, average='binary')
            auc_roc = roc_auc_score(label_classifications, probabilities)
            auc_pr = average_precision_score(label_classifications, probabilities)
        except ValueError:  # Handles single class case
            if np.all(label_classifications == 1):
                logloss = -np.log(np.mean(probabilities))  # Loss if all true positives
            else:
                logloss = -np.log(1 - np.mean(probabilities))  # Loss if all true negatives
            accuracy = 1.0
            precision_by_class = np.array([1.0, 0.0])  # or [0.0, 1.0] depending on which class is present
            recall_by_class = np.array([1.0, 0.0])
            f1 = 1.0  # or 0.0 depending on context
            auc_roc = 0.5
            auc_pr = 1.0 if np.all(label_classifications == 1) else 0.0
            
        #primary = ('rmse', rmse)
        primary = ('logloss', logloss)

        logger.info(f"Count {reg_custom_metric.count} (tgt threshold={tgt_threshold:.4f}) (pred threshold={pred_threshold:.4f}): "
                    f"MSE: {mse:.5f}, RMSE: {rmse:.5f}, Cross Entropy: {crossentropy:.5f}, "
                    f"R2: {r2:.5f}, "
                    f"(Imputed) Precision (incorrect/0): {precision_by_class[0]:.5f}, "
                    f"(Imputed) Precision (correct/1): {precision_by_class[1]:.5f}, "
                    f"(Imputed) Recall (incorrect/0): {recall_by_class[0]:.5f}, "
                    f"(Imputed) Recall (correct/1): {recall_by_class[1]:.5f}, "
                    f"(Imputed) F1: {f1}, "
                    f"(Imputed) Accuracy: {accuracy:.5f}, (Imputed) AUC ROC: {auc_roc:.5f}, (Imputed) AUC PR: {auc_pr:.5f}")
        
    else:
        logger.info(f"[{reg_custom_metric.count}] -{primary[0]}: {primary[1]:.5f}")

    if DirectCall:
        return [('MSE', mse, 'RMSE', rmse, 'Cross Entropy', crossentropy, 'R2', r2,

                 '(Imputed) Precision (incorrect/0)', precision_by_class[0], 
                 '(Imputed) Precision (correct/1)', precision_by_class[1],
                 '(Imputed) Recall (incorrect/0)', recall_by_class[0],
                 '(Imputed) Recall (correct/1)', recall_by_class[1],
                 '(Imputed) F1', f1, 
                 '(Imputed) Accuracy', accuracy, '(Imputed) AUC ROC', auc_roc, '(Imputed) AUC PR', auc_pr)]
    else:
        return [primary]


def binary_custom_metric(preds, dtrain, triggerFull=False, DirectCall=False, tgt_threshold=None, pred_threshold=0.5, label=None):
    #triggerFull: directly call this func and calculate everything
    #tgt_,pred_threshold: empties to just match reg_custom_metric; tgt_ never used
    #label = set this directly to target values to use this function & ignore dmatrix
    if label is None:
        labels = dtrain.get_label()    
    else:
        labels=label[:]
        
    #filter both based on nan predictions
    valid_mask = ~np.isnan(preds)
    preds = preds[valid_mask]
    labels = labels[valid_mask]
        
    # Always calculate the primary metric
    # Classification metrics
    preds_proba = 1.0 / (1.0 + np.exp(-preds))
    logloss = log_loss(labels, preds_proba)
    primary = ('logloss', logloss)
    if not hasattr(binary_custom_metric, 'count'):
        binary_custom_metric.count = 0
    else:  
        binary_custom_metric.count += 1
    # Determine if this is a train or test call
    #is_train = (binary_custom_metric.count % 2 == 0)
    #iteration = binary_custom_metric.count // 2
    # Log the primary metric for every iteration
    #set_type = "Train" if is_train else "Test"
    # Log all metrics every 10 iterations
    #if ((iteration % 10 == 0) or (triggerFull)):
    if True:
        log_odds = preds  # XGBoost returns log-odds for binary classification
        probabilities = 1 / (1 + np.exp(-log_odds))
        classifications = (probabilities > pred_threshold).astype(int)
        raw_score = probabilities  # For binary, we can use probabilities as raw score
        mse = mean_squared_error(labels, raw_score)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, raw_score)
        r2 = r2_score(labels, raw_score)
        
        try:
            logloss = log_loss(labels, probabilities)
            accuracy = accuracy_score(labels, classifications)
            precision_by_class = precision_score(labels, classifications, average=None, zero_division=0)
            recall_by_class = recall_score(labels, classifications, average=None)
            f1 = f1_score(labels, classifications, average='binary')
            auc_roc = roc_auc_score(labels, probabilities)
            auc_pr = average_precision_score(labels, probabilities)
        except ValueError:  # Handles single class case
            if np.all(label_classifications == 1):
                logloss = -np.log(np.mean(probabilities))  # Loss if all true positives
            else:
                logloss = -np.log(1 - np.mean(probabilities))  # Loss if all true negatives
            accuracy = 1.0
            precision_by_class = np.array([1.0, 0.0])  # or [0.0, 1.0] depending on which class is present
            recall_by_class = np.array([1.0, 0.0])
            f1 = 1.0  # or 0.0 depending on context
            auc_roc = 0.5
            auc_pr = 1.0 if np.all(label_classifications == 1) else 0.0
            
         # Add null checks before formatting
        tgt_str = f"{tgt_threshold:.4f}" if tgt_threshold is not None else "None"
        pred_str = f"{pred_threshold:.4f}" if pred_threshold is not None else "None"
        
        logger.info(f"Count {binary_custom_metric.count} (tgt threshold={tgt_str}) (pred threshold={pred_str}): "
                   f"MSE: {mse:.5f}, RMSE: {rmse:.5f}, LogLoss: {logloss:.5f}, "
                   f"R2: {r2:.5f}, Precision (incorrect/0): {precision_by_class[0]:.5f}, "
                   f"Precision (correct/1): {precision_by_class[1]:.5f}, "
                   f"Recall (incorrect/0): {recall_by_class[0]:.5f}, "
                   f"Recall (correct/1): {recall_by_class[1]:.5f}, "
                   f"F1: {f1:.5f}, Accuracy: {accuracy:.5f}, AUC ROC: {auc_roc:.5f}, AUC PR: {auc_pr:.5f}")
    else:
        logger.info(f"[{binary_custom_metric.count}] -{primary[0]}: {primary[1]:.5f}")
    if DirectCall:
        return [('MSE', mse, 'RMSE', rmse, 'LogLoss', logloss, 'R2', r2, 
                'Precision (incorrect/0)', precision_by_class[0], 
                'Precision (correct/1)', precision_by_class[1],
                'Recall (incorrect/0)', recall_by_class[0],
                'Recall (correct/1)', recall_by_class[1],
                'F1', f1, 'Accuracy', accuracy, 
                'AUC ROC', auc_roc, 'AUC PR', auc_pr)]
    else:
        return [primary]
    
    
def f1_objective_binary(predt: np.ndarray, dtrain) -> tuple:
    """
    Custom F1 objective for binary classification
    """
    y = dtrain.get_label()
    eps = 1e-7
    
    # Apply sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-predt))
    
    # Calculate metrics
    tp = np.sum(y * sigmoid_pred)
    fp = np.sum((1 - y) * sigmoid_pred)
    fn = np.sum(y * (1 - sigmoid_pred))
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    # Modified gradient calculation
    d_sigmoid = sigmoid_pred * (1 - sigmoid_pred)  # derivative of sigmoid
    d_f1 = 2 * (recall * y * (tp + fp) - precision * y * (tp + fn)) / ((precision + recall) ** 2 * (tp + fp) * (tp + fn) + eps)
    grad = d_f1 * d_sigmoid
    
    # More stable hessian
    hess = np.abs(grad) * d_sigmoid + eps
    
    return grad, hess

# For regression, you would need to modify the objective:
def f1_objective_regression(predt: np.ndarray, dtrain) -> tuple:
    """
    Custom F1 objective for regression
    Converts regression problem to binary classification using a threshold
    """
    y = dtrain.get_label()
    threshold = 0.5
    eps = 1e-7
    
    # Smooth threshold using sigmoid
    sigmoid = lambda x: 1 / (1 + np.exp(-10 * (x - threshold)))
    smooth_predt = sigmoid(predt)
    smooth_y = sigmoid(y)
    
    # Calculate smooth F1 components
    tp = np.sum(smooth_y * smooth_predt)
    fp = np.sum((1 - smooth_y) * smooth_predt)
    fn = np.sum(smooth_y * (1 - smooth_predt))
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    # Gradient through smooth functions
    d_sigmoid = smooth_predt * (1 - smooth_predt) * 10  # chain rule through sigmoid
    d_f1 = 2 * (recall * smooth_y * (tp + fp) - precision * smooth_y * (tp + fn)) / ((precision + recall) ** 2 * (tp + fp) * (tp + fn) + eps)
    grad = d_f1 * d_sigmoid
    
    # More informative hessian
    hess = np.abs(grad) * d_sigmoid + eps
    
    return grad, hess

    
#parses string for metrics...bc its in a string format for some reasonm
import re
def parse_eval_result(eval_result):
    """Parse the evaluation result string and return all metrics and their values."""
    pattern = r'(\w+[-\w+]*):([-+]?[0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, eval_result)
    return {metric: float(value) for metric, value in matches}

    
def callback(env):
    if env.iteration % 10 == 0:
        msg = f'[{env.iteration}] '
        for metric in env.evaluation_result_list:
            msg += f'{metric[0]}: {metric[1]:.5f}, '
        #print(msg)
        logger.info(msg.strip(', '))        
        
        
        
def create_dmatrix(file, features, target, scaler=None):
    df = pd.read_parquet(file)
    X = df[features].values
    y = df[target].values
    if scaler:
        X = scaler.transform(X)
    return DMatrix(X, label=y, feature_names=features)

def setup_logger(log_file):
    #os.makedirs(os.path.dirname(log_file), exist_ok=True)
    #absolute_path = os.path.abspath(log_file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    

def create_file_list(files, output_file, features, target):
    with open(output_file, 'w') as f:
        for file in files:
            f.write(f"{file}\n")

def preprocess_dataframe(df, preserve_columns=['SKILL']):
    for col in df.columns:
        if col not in preserve_columns:  # Skip conversion for preserved columns
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].astype(float)
    return df

def load_parquet_files(file_list, features, target, CorrectnessBinary, TestMode):
    dfs = []
    if CorrectnessBinary:
        logger.info("Binarizing target as we load...")
    if TestMode:
        logger.info("Test Mode...running only a couple files.")
        file_list=file_list[:2]
    
    for file in file_list:
        df = pd.read_parquet(file, columns=features + [target])
        df = preprocess_dataframe(df)
        if CorrectnessBinary:
            df[target] = (df[target] // 1).clip(0, 1).astype('int8')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_single_parquet_file(file, features, target, CorrectnessBinary):
    if CorrectnessBinary:
        logger.info("Binarizing target as we load...")
    
    df = pd.read_parquet(file, columns=features + [target])
    df = preprocess_dataframe(df)
    if CorrectnessBinary:
        df[target] = (df[target] // 1).clip(0, 1).astype('int8')
    return df


def process_parquet_file(file, features, target, CorrectnessBinary, scaler, scalerFlag=True, ItemAgnosticFit=False, ItemAgnosticDoubleFit=False):
    # Load a single parquet file
    data = load_single_parquet_file(file, features, target, CorrectnessBinary)
    
    if scalerFlag:
        # Scale the features
        data_scaled = scaler.transform(data[features])
    else:
        data_scaled = data
            
    if ItemAgnosticDoubleFit:
        # Create copy that will have items obliterated
        data_scaled_agnostic = data_scaled.copy()
        erase_columns = ["discriminability", "difficulty", "guessing", "inattention", "discriminability_error", "difficulty_error", "guessing_error", "inattention_error", "auc_roc", "optimal_threshold", "tpr", "tnr", "skill_optimal_threshold", "student_mean_accuracy"]
        data_scaled_agnostic[erase_columns] = np.nan
        
        # Combine both versions
        combined_data = pd.concat([data_scaled, data_scaled_agnostic])
        combined_labels = pd.concat([data_scaled[target], data_scaled[target]])
        dmatrix = xgb.DMatrix(combined_data[features], label=combined_labels)
        
    else:
        if ItemAgnosticFit:
            erase_columns = ["discriminability", "difficulty", "guessing", "inattention", "discriminability_error", "difficulty_error", "guessing_error", "inattention_error", "auc_roc", "optimal_threshold", "tpr", "tnr", "skill_optimal_threshold", "student_mean_accuracy"]
            data_scaled[erase_columns] = np.nan
            
        # Create DMatrix
        dmatrix = xgb.DMatrix(data_scaled[features], label=data[target])
    
    return dmatrix

#More Profiicency Model fit code


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, DMatrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import os
import glob
import json

import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()



class NanPreservingScaler:
    def __init__(self):
        self.scalers = {}
        self.column_means = {}
        self.numeric_columns = None
        self.is_fitted = False

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
                # Fit the scaler if it hasn't been fitted yet
                if not hasattr(self.scalers[col], 'mean_') or not hasattr(self.scalers[col], 'scale_'):
                    self.scalers[col].fit(col_data)
                else:
                    self.scalers[col].partial_fit(col_data)
            
            current_mean = X_numeric[col].mean()
            if col in self.column_means:
                # Update running average
                n = len(X_numeric)
                total = self.column_means[col] * n + current_mean * n
                self.column_means[col] = total / (2 * n)
            else:
                self.column_means[col] = current_mean
        
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("This NanPreservingScaler instance is not fitted yet. Call 'partial_fit' with appropriate arguments before using this estimator.")
            
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        
        X_numeric = X[self.numeric_columns].copy()
        
        for col in self.numeric_columns:
            col_data = X_numeric[col].values.reshape(-1, 1)
            nan_mask = np.isnan(col_data)
            
            # Fill NaN values with column mean
            non_nan_data = col_data.copy()
            non_nan_data[nan_mask] = self.column_means[col]
            
            # Transform the data
            transformed_data = self.scalers[col].transform(non_nan_data)
            
            # Restore NaN values
            transformed_data[nan_mask] = np.nan
            
            X_numeric[col] = transformed_data.flatten()
        
        X.loc[:, self.numeric_columns] = X_numeric
        return X

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)    
            

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import scipy.special

class SigmoidXGBRegressor:
    """XGBoost regressor with sigmoid transformation for semi-binary targets."""
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs
        
    @property
    def feature_names(self):
        """Expose the underlying model's feature names"""
        return self.model.feature_names
    
    def train(self, dtrain, num_boost_round=100, evals=None, early_stopping_rounds=None, custom_metric=None):
        # Transform target values before training
        y = dtrain.get_label()
        y_transformed = scipy.special.logit(np.clip(y, 1e-15, 1 - 1e-15))
        dtrain.set_label(y_transformed)
        
        if evals:
            # Transform validation set labels too
            new_evals = []
            for dmat, name in evals:
                y_eval = dmat.get_label()
                y_eval_transformed = scipy.special.logit(np.clip(y_eval, 1e-15, 1 - 1e-15))
                dmat.set_label(y_eval_transformed)
                new_evals.append((dmat, name))
        else:
            new_evals = None

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=new_evals,
            early_stopping_rounds=early_stopping_rounds,
            custom_metric=custom_metric
        )
        return self

    def predict(self, dmatrix):
        raw_preds = self.model.predict(dmatrix)
        return scipy.special.expit(raw_preds)  # sigmoid transform predictions

    def save_model(self, filename):
        self.model.save_model(filename)
        
    def load_model(self, filename):
        self.model = xgb.Booster()
        self.model.load_model(filename)            
    

#used in SigmoidTransformOutput
#fits scaler after model fit ('score' --> 'probability') using cbe
class IncrementalSigmoidFitter:
    def __init__(self):
        self.sum_pred_true = 0
        self.sum_pred_false = 0
        self.count_true = 0
        self.count_false = 0
        
    def accumulate(self, predictions, targets):
        #mask_true = targets == 1
        mask_true = targets > 0.5
        mask_false = ~mask_true
        self.sum_pred_true += np.sum(predictions[mask_true])
        self.sum_pred_false += np.sum(predictions[mask_false])
        self.count_true += np.sum(mask_true)
        self.count_false += np.sum(mask_false)
        
    def fit(self):
        from scipy.optimize import minimize

        mean_pred_true = self.sum_pred_true / (self.count_true + 1e-7)  
        mean_pred_false = self.sum_pred_false / (self.count_false + 1e-7)

        def cbe_loss(params):
            a, b = params
            p_true = 1 / (1 + np.exp(-a * (mean_pred_true - b)))  # b is the centering point
            p_false = 1 / (1 + np.exp(-a * (mean_pred_false - b)))        
            return -(self.count_true * np.log(p_true + 1e-7) + 
                    self.count_false * np.log(1 - p_false + 1e-7))
        result = minimize(cbe_loss, x0=[1.0, 0.5], method='L-BFGS-B', 
                         bounds=[(0, 100), (0.01, 0.99)])
        return result.x        
        

def train_proficiency_model(input_folder, model_output_folder, CorrectnessBinary, TestMode=False, n_estimators=200,
                        scalerFlag = False, early_stopping_rounds=10, memorySaver = True, UseSigmoidTransform = False,
                        SigmoidTransformOutput=False, F1ObjectiveMeasure=False, ItemAgnosticFit=False, ItemAgnosticDoubleFit=False, FUTURE_WINDOW_TRAINING=0):
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)

    # Set up logging
    log_file = os.path.join(model_output_folder, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    setup_logger(log_file)
    logger.info(f"Starting model training. Logs will be saved to {log_file}")

    # Load column names
    column_names_file = os.path.join(input_folder, 'feature_engineered_column_names.json')
    logger.info(f"Loading column names from {column_names_file}")
    with open(column_names_file, 'r') as f:
        column_names = json.load(f)

    # Define target 
    target = 'CORRECTNESS'
    
    # Define features
    # Define the exclusion list
    exclusion_list = [
        'STUDENTID', 'SKILL', 'QUESTIONID', 'CORRECTNESS', 'DURATIONSECONDS',
        'ANSWERPOSITIONINSESSION', 'EVENT_RANK'
    ]

    # Function to check if a column name starts with 'QUESTIONID_LAG_'
    def is_questionid_lag(column_name):
        return column_name.startswith('QUESTIONID_LAG_')

    # Function to check if a column is future correctness
    def is_future_correctness(column_name):
        return column_name.startswith('FUTURE_CORRECTNESS_')
    
    # Filter out the exclusion list and QUESTIONID_LAG columns
    model_input_features = [
        col for col in column_names 
        if col not in exclusion_list and not is_questionid_lag(col)
        and not is_future_correctness(col)
    ]
    features = model_input_features

    # Print the results
    logger.info("Excluded columns:")
    logger.info([col for col in exclusion_list if col in column_names])
    logger.info("\nExcluded QUESTIONID_LAG columns:")
    logger.info([col for col in column_names if is_questionid_lag(col)])
    logger.info("\nModel input features:")
    logger.info(model_input_features)
    

    
    # Save feature names
    feature_filename = os.path.join(model_output_folder, 'feature_names.json')
    with open(feature_filename, 'w') as f:
        json.dump(features, f)
    logger.info(f"Feature names saved to {feature_filename}")

    # Get all parquet files
    all_files = glob.glob(os.path.join(input_folder, "*.parquet"))
      
    if TestMode:
        logger.info("Test Mode: Finding files with sufficient data...")
        valid_files = []
        min_rows = 10  # Minimum rows needed for training

        for file in all_files:
            # Load and check data size
            df = pd.read_parquet(file)
            if len(df) >= min_rows:
                valid_files.append(file)
                logger.info(f"Found valid file: {file} with {len(df)} rows")
                if len(valid_files) == 2:  # We found enough files
                    break

        if len(valid_files) < 2:
            raise ValueError(f"Could not find 2 files with at least {min_rows} rows each. Found {len(valid_files)} valid files.")
        all_files = valid_files
        logger.info(f"Test mode will use these files: {valid_files}")

        #split into train and test
        train_files, test_files = train_test_split(all_files, test_size=0.5, random_state=42)
    else:
        # Split files into train and test
        train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)


    #new code to handle too much data
    if scalerFlag:
        logger.info("Initializing scaler...")
        scaler = NanPreservingScaler()

        # Fit the scaler on all training data
        logger.info("Fitting scaler on all training data...")
        for file in train_files:
            data = load_single_parquet_file(file, features, target, CorrectnessBinary)
            scaler = scaler.partial_fit(data[features])

        # Save the scaler
        scaler_filename = os.path.join(model_output_folder, 'nan_preserving_scaler_proficiency.joblib')
        joblib.dump(scaler, scaler_filename)
        logger.info(f"NanPreservingScaler saved to {scaler_filename}")
    else:
        scaler = None

    if UseSigmoidTransform:
        logger.info("Using sigmoid-transformed XGBoost...")
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['logloss', 'rmse'],
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'nthread': 8
        }
        model = SigmoidXGBRegressor(**params).train
        custom_metric = reg_custom_metric if not CorrectnessBinary else binary_custom_metric    
    else:
        logger.info("Using standard XGBoost...")
        params = {
#            'objective': 'reg:squarederror' if not CorrectnessBinary else 'binary:logistic',
            'objective': 'reg:logistic' if not CorrectnessBinary else 'binary:logistic',
#            'eval_metric': 'rmse' if not CorrectnessBinary else 'logloss',
            'eval_metric': ['logloss', 'rmse'],
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'nthread': 8
        }
        model = xgb.train
        custom_metric = reg_custom_metric if not CorrectnessBinary else binary_custom_metric    
        
    # Set XGBoost parameters
    if CorrectnessBinary:
        logger.info("Using logistic, logloss...")        
        metric = 'logloss'
        primary_metric = 'test-eval-logloss'
        custom_metric = binary_custom_metric
    else:
#             logger.info("Using regressor, rmse, reg:squarederror...")
#            metric = 'rmse'
        metric = 'logloss'
        primary_metric = 'test-eval-logloss'
        custom_metric = reg_custom_metric
    logger.info(f"Using regressor,{params['eval_metric']}, {params['objective']}...")


    if F1ObjectiveMeasure:
        if CorrectnessBinary:
            logger.info("Using F1 binary for objective")
        else:
            logger.info("Using F1 regression for objective")
        
    # Train the model incrementally
    logger.info("Starting XGBoost training...")
    model = None
    if TestMode:
        n_estimators = 5

    from collections import defaultdict

    #primary_metric = None
    best_score = float('inf')  # Initialize with positive infinity for metrics where lower is better
    best_iteration = 0
    no_improve_count = 0

    mean_threshold_tgt=[]
    median_threshold_pred=[]    
    
    for epoch in range(n_estimators):
        epoch_scores = defaultdict(float)
        for i, train_file in enumerate(train_files):
            
            dtrain = process_parquet_file(train_file, features, target, CorrectnessBinary, scaler, scalerFlag, ItemAgnosticFit, ItemAgnosticDoubleFit)
            
#             dtrain = process_parquet_file(train_file, features, target, CorrectnessBinary, scaler, scalerFlag, ItemAgnosticFit)
            #dtrain = process_parquet_file(train_file, features, target, CorrectnessBinary, scaler)
            # Process a test file
            test_file = test_files[i % len(test_files)]  # Cycle through test files
            #dtest = process_parquet_file(test_file, features, target, CorrectnessBinary, scaler)
#             dtest = process_parquet_file(test_file, features, target, CorrectnessBinary, scaler, scalerFlag, ItemAgnosticFit)

            dtest = process_parquet_file(test_file, features, target, CorrectnessBinary, scaler, scalerFlag, ItemAgnosticFit, ItemAgnosticDoubleFit)

            # XGBoost training code (unchanged)
            if model is None:
                if F1ObjectiveMeasure:
                    params['objective'] = None  # Remove built-in objective
                    if CorrectnessBinary:
                        model = xgb.train(
                            params,
                            dtrain,
                            num_boost_round=1,
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            custom_metric=custom_metric,
                            obj=f1_objective_binary
                        )
                    else:
                        model = xgb.train(
                            params,
                            dtrain,
                            num_boost_round=1,
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            custom_metric=custom_metric,
                            obj=f1_objective_regression
                        )
                else: #no F1
                    model = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=1,
                        evals=[(dtrain, 'train'), (dtest, 'test')],
                        custom_metric=custom_metric)
                    
            else: #non-first run
                if F1ObjectiveMeasure:
                    params['objective'] = None  # Remove built-in objective
                    if CorrectnessBinary:
                        model = xgb.train(
                            params,
                            dtrain,
                            num_boost_round=1,
                            xgb_model=model,
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            custom_metric=custom_metric,
                            obj=f1_objective_binary
                        )
                    else:
                        model = xgb.train(
                            params,
                            dtrain,
                            num_boost_round=1,
                            xgb_model=model,
                            evals=[(dtrain, 'train'), (dtest, 'test')],
                            custom_metric=custom_metric,
                            obj=f1_objective_regression
                        )
                else:
                    model = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=1,
                        xgb_model=model,
                        evals=[(dtrain, 'train'), (dtest, 'test')],
                        custom_metric=custom_metric)

            # Get the evaluation scores
            train_scores = parse_eval_result(model.eval(dtrain))
            test_scores = parse_eval_result(model.eval(dtest))

            #accumulate median points (for test metric loop later)
            mean_threshold_tgt.append(np.mean(dtrain.get_label()))
            median_threshold_pred.append(np.median(model.predict(dtrain)))

            
            # Accumulate scores
            for metric, score in train_scores.items():
                epoch_scores[f'train-{metric}'] += score
            for metric, score in test_scores.items():
                epoch_scores[f'test-{metric}'] += score

            # Log all metrics
            log_message = f"Epoch {epoch+1}/{n_estimators}, File {i+1}/{len(train_files)}"
            for metric, score in {**train_scores, **test_scores}.items():
                log_message += f", {metric}: {score:.4f}"
            logger.info(log_message)

        # Calculate average scores for the epoch
        avg_scores = {metric: score / len(train_files) for metric, score in epoch_scores.items()}

        # Backup the model
        model_filename = os.path.join(model_output_folder, 'xgb_model.json')
        model.save_model(model_filename)
        logger.info(f"Backing up model training progress. Saved to {model_filename}")

        # Set primary metric if not set
        if primary_metric is None:
            if CorrectnessBinary:
                # Assuming we want to use a test metric, and 'eval-rmse' is our target metric
                primary_metric = next((metric for metric in avg_scores if metric.startswith('test-') and 'eval-logloss' in metric), None)                
            else:
                # Assuming we want to use a test metric, and 'eval-rmse' is our target metric
                primary_metric = next((metric for metric in avg_scores if metric.startswith('test-') and 'eval-rmse' in metric), None)
            if primary_metric is None:
                raise ValueError("Could not determine primary metric from available metrics")

        # Log average scores
        log_message = f"Epoch {epoch+1} complete."
        for metric, score in avg_scores.items():
            log_message += f" Avg {metric}: {score:.4f}"
        logger.info(log_message)

        # Check for early stopping (using the primary test metric)
        if avg_scores[primary_metric] < best_score:
            best_score = avg_scores[primary_metric]
            best_iteration = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_rounds:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.info(f"Best iteration: {best_iteration+1}, Best {primary_metric}: {best_score:.4f}")

    logger.info(f"Model training completed.")
    logger.info('\n'.join([f"{feature}: {importance:.4f}" for feature, importance in sorted(zip(model.feature_names, model.get_score(importance_type='gain').values()), key=lambda x: x[1], reverse=True)]))



    # Final evaluation
    from collections import defaultdict
    logger.info("Final model metrics.")
    metric_sums = defaultdict(float)
    num_test_files = len(test_files)

    mean_threshold_tgt=0.5
    #mean_threshold_tgt=np.mean(mean_threshold_tgt)
    median_threshold_pred=np.mean(median_threshold_pred)
    logger.info(f"thresholds tgt, pred: {mean_threshold_tgt}, {median_threshold_pred}")
    
    for i, test_file in enumerate(test_files):
        #dtest = process_parquet_file(test_file, features, target, CorrectnessBinary, scaler)
        dtest = process_parquet_file(test_file, features, target, CorrectnessBinary, scaler, scalerFlag)
        logger.info(f"Test set {i+1}:")
        metrics = custom_metric(model.predict(dtest), dtest, True, True, tgt_threshold=mean_threshold_tgt, pred_threshold=median_threshold_pred)[0]  # Assuming custom_metric returns a list with one tuple

        # Aggregate metrics
        for j in range(0, len(metrics), 2):
            metric_name, metric_value = metrics[j], metrics[j+1]
            metric_sums[metric_name] += metric_value

    # Calculate and log average metrics
    logger.info("Average metrics across all test sets:")
    for metric_name, metric_sum in metric_sums.items():
        avg_metric = metric_sum / num_test_files
        logger.info(f"{metric_name}: {avg_metric:.5f}")

    # Save the model
    model_filename = os.path.join(model_output_folder, 'xgb_model.json')
    model.save_model(model_filename)
    logger.info(f"Model saved to {model_filename}")



#"skill" model -- trains sigmoid on item prediction
#actual inference wrapper currently runs for all items in skill and aggregates
# overly centralized predictions, (probably due to boundary effects), 
#sigmoid to push 'scores' into probabilities and closer to 0 or 1
#also using cross entropy to help that
def train_skill_model(input_folder, model_output_folder, proficiency_model_file, CorrectnessBinary,
                     TestMode=False, FUTURE_WINDOW=0, scalerFlag=False,
                     UseSigmoidTransform = True, ItemAgnosticFit=False):
    """
    Trains a skill model that fits a sigmoid transformation to proficiency scores,
    using future performance metrics as targets.
    """
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)

    # Set up logging
    log_file = os.path.join(model_output_folder, f'skill_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    setup_logger(log_file)
    logger.info(f"Starting skill model training. Logs will be saved to {log_file}")

       
    # Load the proficiency model
    if UseSigmoidTransform:
        logger.info("Loading sigmoid-transformed proficiency model...")
        proficiency_model = SigmoidXGBRegressor()
        proficiency_model.load_model(proficiency_model_file)
    else:
        proficiency_model = xgb.Booster()
        proficiency_model.load_model(proficiency_model_file)   

    #grab features from model directly
    proficiency_features = proficiency_model.feature_names    
    # Define target for first model
    logger.info(f"First model uses {len(proficiency_features)} features")

    
    # Load scaler if needed
    if scalerFlag:
        proficiency_scaler_filename = os.path.join(os.path.dirname(proficiency_model_file), 
                                                 'nan_preserving_scaler_proficiency.joblib')
        proficiency_scaler = joblib.load(proficiency_scaler_filename)
        logger.info(f"Loaded NanPreservingScaler from {proficiency_scaler_filename}")
    else:
        proficiency_scaler = None

    # Get all parquet files
    all_files = glob.glob(os.path.join(input_folder, "*.parquet"))
    
    if TestMode:
        logger.info("Test Mode: Finding files with sufficient data...")
        valid_files = []
        min_rows = 10  # Minimum rows needed for training

        for file in all_files:
            df = pd.read_parquet(file)
            if len(df) >= min_rows:
                valid_files.append(file)
                logger.info(f"Found valid file: {file} with {len(df)} rows")
                if len(valid_files) == 2:  # We found enough files
                    break

        if len(valid_files) < 2:
            raise ValueError(f"Could not find 2 files with at least {min_rows} rows each.")
        all_files = valid_files
        logger.info(f"Test mode will use these files: {valid_files}")

    # Split files into train and test
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    logger.info(f"Fitting sigmoid transformation against future window of {FUTURE_WINDOW}...")
    logger.info(f"Note: We use the unclipped predictions to fit the sigmoid transformer (even though Proficiency Model fitting, as of now, uses clipped predictions to drive the loss function.")
    
    # Initialize arrays for all predictions and targets
    all_predictions = []
    all_targets = []

    for i, train_file in enumerate(train_files):
        # Load features including future correctness if needed
        loaded_features = proficiency_features[:]
        if FUTURE_WINDOW > 0:
            loaded_features += [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]

        # Load and prepare data
        train_data = load_single_parquet_file(train_file, loaded_features, 'CORRECTNESS', CorrectnessBinary)
        
        if scalerFlag:
            train_data_scaled = proficiency_scaler.transform(train_data)
        else:
            train_data_scaled = train_data

        #obliterate next-item information
        if ItemAgnosticFit:
            logger.info(f"Erasing next-item information.")
            erase_columns = ["discriminability", "difficulty", "guessing", "inattention", "discriminability_error", "difficulty_error", "guessing_error", "inattention_error", "auc_roc", "optimal_threshold", "tpr", "tnr", "skill_optimal_threshold", "student_mean_accuracy"]
            # Set specified columns to NaN
            train_data_scaled[erase_columns] = np.nan
            
        # Get proficiency model predictions
        predictions = proficiency_model.predict(xgb.DMatrix(train_data_scaled[proficiency_features]))
        all_predictions.extend(predictions)

        # Calculate target as mean of future correctness
        if FUTURE_WINDOW == 0:
            targets = train_data_scaled['CORRECTNESS']
        else:
            future_cols = [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]
            targets = train_data_scaled[['CORRECTNESS'] + future_cols].mean(axis=1)
        
        all_targets.extend(targets)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    logger.info(
        f'Predictions stats: '
        f'min={np.nanmin(all_predictions):.3f}, '
        f'max={np.nanmax(all_predictions):.3f}, '
        f'mean={np.nanmean(all_predictions):.3f}, '
        f'nan_count={np.isnan(all_predictions).sum()} | '
        f'Targets stats: '
        f'min={np.nanmin(all_targets):.3f}, '
        f'max={np.nanmax(all_targets):.3f}, '
        f'mean={np.nanmean(all_targets):.3f}, '
        f'nan_count={np.isnan(all_targets).sum()}'
    )
    
    # Fit sigmoid parameters (a and b) using optimization
    from scipy.optimize import minimize

    def sigmoid_loss(params):
        a, b = params
        sigmoid_pred = 1 / (1 + np.exp(-a * (all_predictions - b)))
        return cross_entropy(all_targets, sigmoid_pred)
        # return np.mean((sigmoid_pred - all_targets) ** 2)  # MSE version

    # Initial guess for parameters
    initial_guess = [1.0, 0.5]
    
    # Optimize parameters
    result = minimize(sigmoid_loss, initial_guess, method='Nelder-Mead')
    a, b = result.x

    logger.info(f"Fitted sigmoid parameters: a={a:.3f}, b={b:.3f}")
    
    # Save parameters
    params_dict = {
        'a': float(a),
        'b': float(b),
        'future_window': FUTURE_WINDOW
    }
    
    params_file = os.path.join(model_output_folder, 'sigmoid_params.json')
    with open(params_file, 'w') as f:
        json.dump(params_dict, f)

    logger.info(f"Saved skill sigmoid parameters to {params_file}")

    bounds = [
    (0, np.inf),  # a can be any real number
    (0, 1)             # b is constrained between 0 and 1
    ]

    # Optimize parameters using L-BFGS-B method which supports bounds
    result = minimize(sigmoid_loss, initial_guess, method='L-BFGS-B', bounds=bounds)
    a, b = result.x
    logger.info(f"(Capped NOT CURRENTLY USED) Fitted sigmoid parameters: a={a:.3f}, b={b:.3f}")

    
    # Validate on test set
    logger.info("Validating skill model on test set...")
    #use this for error metric
    custom_metric = reg_custom_metric

    from collections import defaultdict
    logger.info("Sigmoid Transform fit (proficiency model) metrics.")
    prof_metric_sums = defaultdict(float)
    
    #test_predictions = []
    all_test_targets = []
    all_transformed_predictions = []
    
    for test_file in test_files:
        loaded_features = proficiency_features[:]
        if FUTURE_WINDOW > 0:
            loaded_features += [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]

        test_data = load_single_parquet_file(test_file, loaded_features, 'CORRECTNESS', CorrectnessBinary)
        
        if scalerFlag:
            test_data_scaled = proficiency_scaler.transform(test_data)
        else:
            test_data_scaled = test_data
        
        dtest=xgb.DMatrix(test_data_scaled[proficiency_features])
        predictions = proficiency_model.predict(dtest)
        transformed_predictions = 1 / (1 + np.exp(-a * (predictions - b)))

        if FUTURE_WINDOW == 0:
            targets = test_data_scaled['CORRECTNESS']
        else:
            future_cols = [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]
            targets = test_data_scaled[['CORRECTNESS'] + future_cols].mean(axis=1)

        all_transformed_predictions.extend(transformed_predictions)
        all_test_targets.extend(targets)
        #call custom metrics, uses default of 0.5 to binarize, use 'label' to directly define target data
        transformed_prof_metrics = custom_metric(transformed_predictions, dtest, True, True, label=targets)[0]
        
        # Aggregate metrics as we go
        for j in range(0, len(transformed_prof_metrics), 2):
            prof_metric_name, prof_metric_value = transformed_prof_metrics[j], transformed_prof_metrics[j+1]
            prof_metric_sums[prof_metric_name] += prof_metric_value


    # Calculate and log average metrics
    num_test_files = len(test_files)
    logger.info("Average metrics across all test sets (proficiency model):")
    for prof_metric_name, prof_metric_sum in prof_metric_sums.items():
        prof_avg_metric = prof_metric_sum / num_test_files
        logger.info(f"{prof_metric_name}: {prof_avg_metric:.5f}")
 

    def optimize_proficiency_thresholds(predictions, actual_outcomes, initial_thresholds=(0.5, 0.7)):
        from scipy.optimize import minimize
        import numpy as np

        predictions = np.asarray(predictions, dtype=np.float64)
        actual_outcomes = np.asarray(actual_outcomes, dtype=np.float64)

        def threshold_loss(thresholds):
            struggling_threshold, proficiency_threshold = thresholds

            # Create masks for each category
            low_mask = predictions < struggling_threshold
            mid_mask = (predictions >= struggling_threshold) & (predictions < proficiency_threshold)
            high_mask = predictions >= proficiency_threshold

            # Calculate correctness rates with heavy penalties for empty categories
            if np.sum(low_mask) == 0:
                low_correct = 0.8  # Heavily penalize no students in low category by setting far from target
            else:
                low_correct = np.mean(actual_outcomes[low_mask])

            if np.sum(mid_mask) == 0:
                mid_correct = 0.4  # Penalize empty middle category
            else:
                mid_correct = np.mean(actual_outcomes[mid_mask])

            if np.sum(high_mask) == 0:
                high_correct = 0.5  # Penalize empty high category
            else:
                high_correct = np.mean(actual_outcomes[high_mask])

            # Loss terms for each category's correctness rate
            low_loss = max(0, low_correct - 0.6)**2 * 10  # Penalize if > 0.6
            mid_loss = (mid_correct - 0.7)**2             # Target around 0.7
            high_loss = max(0, 0.8 - high_correct)**2 * 10  # Penalize if < 0.8

            # Additional penalty for empty categories
            empty_penalty = 0
            if np.sum(low_mask) == 0 or np.sum(mid_mask) == 0 or np.sum(high_mask) == 0:
                empty_penalty = 100

            return low_loss + mid_loss + high_loss + empty_penalty

        # Wide bounds to allow finding good thresholds
        bounds = [(0.2, 0.6), (0.6, 0.95)]

        # Minimum gap between thresholds
        constraints = [{
            'type': 'ineq',
            'fun': lambda x: x[1] - x[0] - 0.15  # minimum 0.15 gap
        }]

        try:
            result = minimize(
                threshold_loss,
                x0=initial_thresholds,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'ftol': 1e-6, 'maxiter': 1000}
            )

            print(f"\nOptimization {'succeeded' if result.success else 'failed'}: {result.message}")

            # Print diagnostics
            thresholds = result.x
            low_mask = predictions < thresholds[0]
            mid_mask = (predictions >= thresholds[0]) & (predictions < thresholds[1])
            high_mask = predictions >= thresholds[1]

            print("\nCategory performance:")
            print(f"Struggling (<{thresholds[0]:.3f}): {np.mean(actual_outcomes[low_mask])*100:.1f}% correct ({np.sum(low_mask)} students)")
            print(f"Practicing: {np.mean(actual_outcomes[mid_mask])*100:.1f}% correct ({np.sum(mid_mask)} students)")
            print(f"Proficient (>{thresholds[1]:.3f}): {np.mean(actual_outcomes[high_mask])*100:.1f}% correct ({np.sum(high_mask)} students)")

            return result.x

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return np.array(initial_thresholds)

    # Usage would be:
    optimal_thresholds = optimize_proficiency_thresholds(
        predictions=all_transformed_predictions,
        actual_outcomes=all_test_targets
    )

    # Save thresholds
    threshold_params = {
        'struggling_threshold': float(optimal_thresholds[0]),
        'proficiency_threshold': float(optimal_thresholds[1])
    }
    threshold_file = os.path.join(model_output_folder, 'threshold_params.json')
    with open(threshold_file, 'w') as f:
        json.dump(threshold_params, f)
    logger.info(f"Saved threshold parameters to {threshold_file}")
    logger.info(f"Thresholds - struggling: {threshold_params['struggling_threshold']:.3f}, proficiency: {threshold_params['proficiency_threshold']:.3f}")
    
    
    # Category analysis
    model_struggling_threshold=float(optimal_thresholds[0])
    model_proficiency_threshold=float(optimal_thresholds[1])
    logger.info("\nItem Proficiency Category Analysis:")

    # Convert predictions to numpy array
    all_transformed_predictions = np.array(all_transformed_predictions, dtype=np.float64)
    all_test_targets = np.array(all_test_targets, dtype=np.float64)
    
    # Calculate category counts and stats
    low_mask = all_transformed_predictions < model_struggling_threshold
    mid_mask = (all_transformed_predictions >= model_struggling_threshold) & (all_transformed_predictions < model_proficiency_threshold)
    high_mask = all_transformed_predictions >= model_proficiency_threshold

    # Get counts
    total_predictions = len(all_transformed_predictions)
    low_count = np.sum(low_mask)
    mid_count = np.sum(mid_mask)
    high_count = np.sum(high_mask)

    # Calculate percentages
    low_percent = (low_count / total_predictions) * 100
    mid_percent = (mid_count / total_predictions) * 100
    high_percent = (high_count / total_predictions) * 100

    # Calculate mean correctness for each category
    low_mean = np.mean(all_test_targets[low_mask]) if low_count > 0 else 0
    mid_mean = np.mean(all_test_targets[mid_mask]) if mid_count > 0 else 0
    high_mean = np.mean(all_test_targets[high_mask]) if high_count > 0 else 0

    # Log results
    logger.info(f"\nStruggling Category (< {model_struggling_threshold}):")
    logger.info(f"Count: {low_count} ({low_percent:.1f}%)")
    logger.info(f"Mean Correctness: {low_mean:.3f}")

    logger.info(f"\nPracticing Category (>= {model_struggling_threshold} and < {model_proficiency_threshold}):")
    logger.info(f"Count: {mid_count} ({mid_percent:.1f}%)")
    logger.info(f"Mean Correctness: {mid_mean:.3f}")

    logger.info(f"\nProficient Category (>= {model_proficiency_threshold}):")
    logger.info(f"Count: {high_count} ({high_percent:.1f}%)")
    logger.info(f"Mean Correctness: {high_mean:.3f}")      


# Confidence model
#Confidence SCaler
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

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
        
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


def train_confidence_model(input_folder, model_output_folder, proficiency_model_file, CorrectnessBinary, 
                           TestMode=False, FUTURE_WINDOW=0, n_estimators=200, scalerFlag = False, 
                           early_stopping_rounds = 10, memorySaver = True, UseSigmoidTransform = False,
                           SigmoidTransformOutput=False, ItemAgnosticFit=False):
    
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)

    # Set up logging
    log_file = os.path.join(model_output_folder, f'confidence_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    setup_logger(log_file)
    logger.info(f"Starting confidence model training. Logs will be saved to {log_file}")

    # Load column names
    column_names_file = os.path.join(input_folder, 'feature_engineered_column_names.json')
    logger.info(f"Loading column names from {column_names_file}")
    with open(column_names_file, 'r') as f:
        column_names = json.load(f)
    
    # Get all parquet files
    all_files = glob.glob(os.path.join(input_folder, "*.parquet"))

    logger.info("Creating DMatrix objects...")
    if scalerFlag:
        # Load the scaler used for the first model
        proficiency_scaler_filename = os.path.join(os.path.dirname(proficiency_model_file), 'nan_preserving_scaler_proficiency.joblib')
        proficiency_scaler = joblib.load(proficiency_scaler_filename)
        logger.info(f"Loaded NanPreservingScaler from {proficiency_scaler_filename}")
    else:
        proficiency_scaler = None
    
    
    # Load feature names used by the first model
    #proficiency_model_feature_file = os.path.join(os.path.dirname(proficiency_model_file), 'feature_names.json')
    #with open(proficiency_model_feature_file, 'r') as f:
    #    proficiency_features = json.load(f)
    
    # Load the first XGBoost model
    #proficiency_model = xgb.Booster()
    #proficiency_model.load_model(proficiency_model_file)   
    
    #this is an xgb extension that hasn't worked that well, will try CB error on it
    if UseSigmoidTransform:
        logger.info("Loading sigmoid-transformed proficiency model...")
        proficiency_model = SigmoidXGBRegressor()
        proficiency_model.load_model(proficiency_model_file)
    else:
        proficiency_model = xgb.Booster()
        proficiency_model.load_model(proficiency_model_file)   
    
    #load sigmoid transformer parameters (2 params) to convert prof score prediction to probability
    if SigmoidTransformOutput:
        proficiency_model_sigmoid_transformer = os.path.join(os.path.dirname(proficiency_model_file), 'sigmoid_params.json')

        # To load and use
        with open(proficiency_model_sigmoid_transformer, 'r') as f:
            params = json.load(f)
            sigmoid_params = [params['a'], params['b']] 
        logger.info(f"Loaded Sigmoid Transformer for Proficiency Scores from {proficiency_model_sigmoid_transformer}. Will use this instead of clipping for proficiency model predictions.")

        def apply_sigmoid(predictions, params):
            a, b = params
            return 1 / (1 + np.exp(-a * (predictions - b)))  # b is the centering point
    
    #grab features from model directly
    proficiency_features = proficiency_model.feature_names
    
    # Define target for first model
    proficiency_target = 'CORRECTNESS'

    logger.info(f"First model uses {len(proficiency_features)} features")
    
    
    # Get all parquet files
    all_files = glob.glob(os.path.join(input_folder, "*.parquet"))
    
    if TestMode:
        logger.info("Test Mode: Finding files with sufficient data...")
        valid_files = []
        min_rows = 10  # Minimum rows needed for training

        for file in all_files:
            # Load and check data size
            df = pd.read_parquet(file)
            if len(df) >= min_rows:
                valid_files.append(file)
                logger.info(f"Found valid file: {file} with {len(df)} rows")
                if len(valid_files) == 2:  # We found enough files
                    break

        if len(valid_files) < 2:
            raise ValueError(f"Could not find 2 files with at least {min_rows} rows each. Found {len(valid_files)} valid files.")
        all_files = valid_files
        logger.info(f"Test mode will use these files: {valid_files}")

        #split into train and test
        train_files, test_files = train_test_split(all_files, test_size=0.5, random_state=42)
    else:
        # Split files into train and test
        train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)


    # Prepare features for the confidence model
    confidence_features = proficiency_features + ['proficiency_model_prediction']

    logger.info(f'confidence feature length: {len(confidence_features)}')
    logger.info(f"model features: {confidence_features}")

    logger.info("Starting confidence XGBoost training...")

    logger.info("Using regressor, rmse, reg:squarederror...")
    metric = 'rmse'
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'nthread': 8
    }
    custom_metric = reg_custom_metric

    if TestMode:
        n_estimators = 5


    primary_metric = None
    best_score = float('inf')
    best_iteration = 0
    no_improve_count = 0
    confidence_model = None

    from collections import defaultdict

    median_threshold_tgt=[]
    median_threshold_pred=[]

    for epoch in range(n_estimators):
        epoch_scores = defaultdict(float)
        for i, train_file in enumerate(train_files):
            # Process train file

            loaded_features = proficiency_features[:]
            #if future windowin,  need to load it too
            if FUTURE_WINDOW>0:
                loaded_features += [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]

            #load, scale, prepare for Prof inference
            train_data = load_single_parquet_file(train_file, loaded_features, proficiency_target, 
                                            CorrectnessBinary)                                
            # Process a test file
            test_file = test_files[i % len(test_files)]  # Cycle through test files
            test_data = load_single_parquet_file(test_file, loaded_features, proficiency_target, 
                                            CorrectnessBinary)
            if scalerFlag:
                test_data_scaled = proficiency_scaler.transform(test_data)
                train_data_scaled = proficiency_scaler.transform(train_data)
            else:
                test_data_scaled = test_data
                train_data_scaled =train_data
                
                
            if ItemAgnosticFit:
                logger.info(f"Erasing next-item information.")
                erase_columns = ["discriminability", "difficulty", "guessing", "inattention", "discriminability_error", "difficulty_error", "guessing_error", "inattention_error", "auc_roc", "optimal_threshold", "tpr", "tnr", "skill_optimal_threshold", "student_mean_accuracy"]
                # Set specified columns to NaN
                test_data_scaled[erase_columns] = np.nan
                train_data_scaled[erase_columns] = np.nan
                

            # Add predictions as a new feature
            train_data_scaled['proficiency_model_prediction'] = proficiency_model.predict(xgb.DMatrix(train_data_scaled[proficiency_features]))
            test_data_scaled['proficiency_model_prediction'] = proficiency_model.predict(xgb.DMatrix(test_data_scaled[proficiency_features]))
            
            if SigmoidTransformOutput:
                train_data_scaled['proficiency_model_prediction'] = apply_sigmoid(train_data_scaled['proficiency_model_prediction'], sigmoid_params)
                test_data_scaled['proficiency_model_prediction'] = apply_sigmoid(test_data_scaled['proficiency_model_prediction'], sigmoid_params)
            else:
                train_data_scaled['proficiency_model_prediction'] = np.clip(train_data_scaled['proficiency_model_prediction'], 0, 1)
                test_data_scaled['proficiency_model_prediction'] = np.clip(test_data_scaled['proficiency_model_prediction'], 0, 1)

            # Calculate the error (difference between prediction and actual)
#                 train_data_scaled['error'] = np.abs(train_data_scaled['proficiency_model_prediction'] - train_data_scaled[proficiency_target])
#                 test_data_scaled['error'] = np.abs(test_data_scaled['proficiency_model_prediction'] - test_data_scaled[proficiency_target])

            #Trigger future windowing for more granular future performance
            #, otherwise just next/current value drives conf model
            if FUTURE_WINDOW == 0:
                proficiency_target = 'CORRECTNESS'
                # Original single-target calculation
                train_data_scaled['error'] = np.abs(train_data_scaled['proficiency_model_prediction'] - train_data_scaled[proficiency_target])
                test_data_scaled['error'] = np.abs(test_data_scaled['proficiency_model_prediction'] - test_data_scaled[proficiency_target])
            else:
                # Multi-target calculation - mean first, then subtract
                logger.info("Using future window of "+str(FUTURE_WINDOW)+" for confidence model inference metrics.")
                targets = ['CORRECTNESS'] + [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]
                train_data_scaled['error'] = np.abs(train_data_scaled['proficiency_model_prediction'] - train_data_scaled[targets].mean(axis=1))
                test_data_scaled['error'] = np.abs(test_data_scaled['proficiency_model_prediction'] - test_data_scaled[targets].mean(axis=1))


            dtrain = xgb.DMatrix(train_data_scaled[confidence_features], label=train_data_scaled['error'])
            dtest = xgb.DMatrix(test_data_scaled[confidence_features], label=test_data_scaled['error'])


            # Train the model
            if confidence_model is None:
                confidence_model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1,
                    evals=[(dtrain, 'train'), (dtest, 'test')],
                    custom_metric=reg_custom_metric
                )
            else:
                confidence_model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1,
                    xgb_model=confidence_model,
                    evals=[(dtrain, 'train'), (dtest, 'test')],
                    custom_metric=reg_custom_metric
                )

            # Get evaluation scores
            train_scores = parse_eval_result(confidence_model.eval(dtrain))
            test_scores = parse_eval_result(confidence_model.eval(dtest))

            #aggregation loop for median split used in test loop later - remove if memory issues
            predictions = confidence_model.predict(dtest)
            targets = dtest.get_label()
            #update median within loop
            median_threshold_tgt.append(np.median(np.array(targets)))
            median_threshold_pred.append(np.median(np.array(predictions)))

            # Accumulate scores
            for metric, score in train_scores.items():
                epoch_scores[f'train-{metric}'] += score
            for metric, score in test_scores.items():
                epoch_scores[f'test-{metric}'] += score

            # Log progress
            log_message = f"Epoch {epoch+1}/{n_estimators}, File {i+1}/{len(train_files)}"
            for metric, score in {**train_scores, **test_scores}.items():
                log_message += f", {metric}: {score:.4f}"
            logger.info(log_message)

        # Calculate average scores for the epoch
        avg_scores = {metric: score / len(train_files) for metric, score in epoch_scores.items()}

        # Set primary metric if not set
        if primary_metric is None:
            # Assuming we want to use a test metric, and 'eval-rmse' is our target metric
            primary_metric = next((metric for metric in avg_scores if metric.startswith('test-') and 'eval-rmse' in metric), None)
            if primary_metric is None:
                raise ValueError("Could not determine primary metric from available metrics")

        # Log average scores
        log_message = f"Epoch {epoch+1} complete."
        for metric, score in avg_scores.items():
            log_message += f" Avg {metric}: {score:.4f}"
        logger.info(log_message)

        # Check for early stopping
        if avg_scores[primary_metric] < best_score:
            best_score = avg_scores[primary_metric]
            best_iteration = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_rounds:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.info(f"Confidence model training completed. Best iteration: {best_iteration+1}, Best {primary_metric}: {best_score:.4f}")

    # Log feature importance
    feature_importance = confidence_model.get_score(importance_type='gain')
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Feature Importance:")
    for feature, importance in sorted_importance:
        logger.info(f"{feature}: {importance:.4f}")



    # Final evaluation
    from collections import defaultdict
    logger.info("Final model metrics.")
    metric_sums = defaultdict(float)
    num_test_files = len(test_files)
    
    #use these from train loop
    median_threshold_tgt = np.mean(median_threshold_tgt)
    median_threshold_pred = np.mean(median_threshold_pred)


    for i, test_file in enumerate(test_files):
        logger.info(f"Test set {i+1}:")

        # Process a test file

        #load, scale, inference
        test_data = load_single_parquet_file(test_file, proficiency_features, proficiency_target, 
                                        CorrectnessBinary)

        if scalerFlag:
            test_data_scaled = proficiency_scaler.transform(test_data)
        else:
            test_data_scaled =test_data

        # Add prediction as a new feature
        #sigmoid transform prediction if True
        if SigmoidTransformOutput:
            test_data_scaled['proficiency_model_prediction'] = apply_sigmoid(proficiency_model.predict(xgb.DMatrix(test_data_scaled[proficiency_features])), sigmoid_params)
        else:
            test_data_scaled['proficiency_model_prediction'] = np.clip(proficiency_model.predict(xgb.DMatrix(test_data_scaled[proficiency_features])),0,1)



        # Calculate the error (difference between prediction and actual)
        test_data_scaled['error'] = np.abs(test_data_scaled['proficiency_model_prediction'] - test_data_scaled[proficiency_target])
        dtest = xgb.DMatrix(test_data_scaled[confidence_features], label=test_data_scaled['error'])   
        
        
        metrics = custom_metric(np.clip(confidence_model.predict(dtest),0,1), dtest, True, True, median_threshold_tgt, median_threshold_pred)[0]  # Assuming custom_metric returns a list with one tuple

        # Aggregate metrics
        for j in range(0, len(metrics), 2):
            metric_name, metric_value = metrics[j], metrics[j+1]
            metric_sums[metric_name] += metric_value

    # Calculate and log average metrics
    logger.info("Average metrics across all test sets:")
    for metric_name, metric_sum in metric_sums.items():
        avg_metric = metric_sum / num_test_files
        logger.info(f"{metric_name}: {avg_metric:.5f}")


        
    # Save the confidence model
    confidence_model_filename = os.path.join(model_output_folder, 'confidence_xgb_model.json')
    confidence_model.save_model(confidence_model_filename)
    logger.info(f"Confidence model saved to {confidence_model_filename}")

    # Save feature names including the new 'first_model_prediction' feature
    feature_filename = os.path.join(model_output_folder, 'confidence_feature_names.json')
    with open(feature_filename, 'w') as f:
        json.dump(confidence_features, f)
    logger.info(f"Confidence model feature names saved to {feature_filename}")

    #'Train' Percentile ranker on Confidence (Error) Predictions
    #scores = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    logger.info(f"Fitting Confidence Score Transformer (percentile ranker)")
    error_predictions = np.clip(confidence_model.predict(dtest),0,1)
    calculator = PercentileRankCalculator(1 - error_predictions)

    # Save the calculator object
    #confidence_model_score_transformer_filename='percentile_rank_calculator.pkl'
    confidence_model_score_transformer_filename = os.path.join(model_output_folder, 'percentile_rank_calculator.pkl')
    calculator.save(confidence_model_score_transformer_filename)
    logger.info(f"Confidence model Score Transformer names saved to {confidence_model_score_transformer_filename}")

    # Load the calculator object
    #loaded_calculator = PercentileRankCalculator.load('percentile_rank_calculator.pkl')

    logger.info(f"The raw probabilities ranks (error_predictions) range from {min(error_predictions):.3f} to {max(error_predictions):.3f}")

    # Use the loaded calculator object (Score Transformer)
    percentile_rank = calculator.get_percentile_rank(1 - error_predictions)
    logger.info(f"The percentile ranks range from {min(percentile_rank):.3f} to {max(percentile_rank):.3f}")
    

    
#inference functions
from datetime import datetime, timezone

def format_datetime(dt):
    """Format datetime to match '2024-12-10T13:34:29.371403527Z' format"""
    if pd.isna(dt):
        return None
    # Convert to UTC if it has timezone info
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    else:
        # If no timezone info, assume UTC
        dt = dt.replace(tzinfo=timezone.utc)
    # Format with 9 decimal places for nanoseconds
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-6] + '000Z'

def transform_row_to_json(row, includeCurrent=True):
    """
    Transform a DataFrame row with lag columns into the required JSON format,
    handling null values by shortening lists.
    """
    # Get the number of lag columns (looking at QUESTIONID_LAG_X columns)
    lag_cols = [col for col in row.index if col.startswith('QUESTIONID_LAG_')]
    num_lags = len(lag_cols)

    # Initialize lists
    question_ids = []
    correctness = []
    duration_seconds = []
    event_times = []

    if includeCurrent:
        # Add current values first (they will become LAG_1 in the API)
        if pd.notna(row['QUESTIONID']):
            question_ids.append(row['QUESTIONID'])
        if pd.notna(row.get('CORRECTNESS')):  # Using get() in case CORRECTNESS doesn't exist
            correctness.append(int(row['CORRECTNESS']))
        if pd.notna(row.get('DURATIONSECONDS')):  # Using get() in case DURATIONSECONDS doesn't exist
            duration_seconds.append(float(row['DURATIONSECONDS']))
        if pd.notna(row['OCCURREDAT']):
            event_times.append(format_datetime(row['OCCURREDAT']))
        
    # Add history values if they're not null
    for i in range(1, num_lags + 1):
        # Question IDs
        question_id = row.get(f'QUESTIONID_LAG_{i}')
        if pd.notna(question_id):
            question_ids.append(question_id)

        # Correctness
        correctness_val = row.get(f'CORRECTNESS_LAG_{i}')
        if pd.notna(correctness_val):
            correctness.append(int(correctness_val))

        # Duration
        duration_val = row.get(f'DURATIONSECONDS_LAG_{i}')
        if pd.notna(duration_val):
            duration_seconds.append(float(duration_val))

        # Event times
        event_time = row.get(f'OCCURREDAT_LAG_{i}')
        if pd.notna(event_time):
            event_times.append(format_datetime(event_time))

    # Create the JSON structure
    return {
        'skillId': row['SKILL'] if pd.notna(row['SKILL']) else None,
        'questionId': row['QUESTIONID'] if pd.notna(row['QUESTIONID']) else None,
        'eventTime': row['OCCURREDAT'] if pd.notna(row['OCCURREDAT']) else None,
        'questionIdsHistory': question_ids,
        'correctnessHistory': correctness,
        'durationSecondsHistory': duration_seconds,
        'eventTimesHistory': event_times
    }

def transform_df_to_json(df):
    """
    Transform entire DataFrame into list of JSON objects.
    """
    return [transform_row_to_json(row) for _, row in df.iterrows()]


    
def add_error_analysis(data_scaled, 
                       item_proficiency_predictions, item_confidence_predictions, item_confidence_scores,
                       item_agnostic_proficiency_predictions, item_agnostic_confidence_predictions, item_agnostic_confidence_scores,
                       skill_proficiency_predictions, skill_confidence_predictions, skill_confidence_scores, 
                       logger, FUTURE_WINDOW=0):
    from scipy.stats import gaussian_kde
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

       
    # Create DataFrames with error metrics for item, item-agnostic, and skill
    df_item = pd.DataFrame({
        'actual': data_scaled['CORRECTNESS'],
        'prof_prediction': item_proficiency_predictions,
        'conf_prediction': item_confidence_predictions,
        'confScore_prediction': (100-item_confidence_scores)/100
    })
    
    df_item_agnostic = pd.DataFrame({
        'actual': data_scaled['CORRECTNESS'],
        'prof_prediction': item_agnostic_proficiency_predictions,
        'conf_prediction': item_agnostic_confidence_predictions,
        'confScore_prediction': (100-item_agnostic_confidence_scores)/100
    })
    
    df_skill = pd.DataFrame({
        'actual': data_scaled['CORRECTNESS'],
        'prof_prediction': skill_proficiency_predictions,
        'conf_prediction': skill_confidence_predictions,
        'confScore_prediction': (100-skill_confidence_scores)/100
    })

    # Calculate error metrics for all three approaches
    for df in [df_item, df_item_agnostic, df_skill]:
        df['model_error'] = df['actual'] - df['prof_prediction']
        df['abs_model_error'] = np.abs(df['model_error'])
        df['conf_error'] = df['abs_model_error'] - df['conf_prediction']
        df['abs_conf_error'] = np.abs(df['conf_error'])
        df['confScore_error'] = df['abs_model_error'] - df['confScore_prediction']
        df['abs_confScore_error'] = np.abs(df['confScore_error'])

    # Create combined model outputs plot (3 rows for item/item-agnostic/skill)
    fig_outputs, axs_outputs = plt.subplots(3, 3, figsize=(15, 15))
    fig_outputs.suptitle('Distributions of Model Outputs - Item vs Item-Agnostic vs Skill Level', y=1.02)
    
    output_variables = [
        ('prof_prediction', 'Prediction'),
        ('confScore_prediction', 'Confidence'),
        ('abs_model_error', 'Error')
    ]
    
    for row, (df, level) in enumerate([
        (df_item, 'Item'), 
        (df_item_agnostic, 'Item-Agnostic'),
        (df_skill, 'Skill')
    ]):
        for col, (var, title) in enumerate(output_variables):
            data = df[var].dropna()
            
            n, bins, patches = axs_outputs[row, col].hist(data, bins=30, density=True, alpha=0.6, 
                                                        color='#6BAED6')
            
            
               # New code with try-except and variance check:
            try:
                # Check if data has enough variance for KDE
                if len(data) > 1 and np.var(data) > 1e-10:  # Adjust threshold as needed
                    
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    axs_outputs[row, col].plot(x_range, kde(x_range), 'r--', lw=1)

                else:
                    # For low variance data, just add a note instead of KDE
                    logger.info(f"Skipping KDE for {level} {title} due to insufficient variance")
                    mean_val = data.mean()
                    axs_outputs[row, col].axvline(mean_val, color='red', linestyle='-', 
                                                 label=f'Mean (data has low variance)')
            except Exception as e:
                logger.info(f"Error in KDE calculation for {level} {title}: {str(e)}")


            mean = data.mean()
            std = data.std()
            n_count = len(data)
            
            axs_outputs[row, col].axvline(mean, color='red', linestyle='--', alpha=0.8,
                                        label=f'Mean: {mean:.3f}\nStd: {std:.3f}\nN: {n_count:,}')
            
            axs_outputs[row, col].set_title(f'{level} {title}')
            axs_outputs[row, col].set_xlabel(title)
            axs_outputs[row, col].set_ylabel('Density')
            axs_outputs[row, col].legend(fontsize=8)
            axs_outputs[row, col].grid(True, alpha=0.3)
            axs_outputs[row, col].set_ylim(bottom=0)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.tight_layout()
    plt.savefig(f'model_output_distributions_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create combined error distribution plots (now 6 rows for item/item-agnostic/skill)
    fig, axs = plt.subplots(6, 3, figsize=(24, 48))
    fig.suptitle('Distribution of Model and Confidence Errors - Item vs Item-Agnostic vs Skill Level', fontsize=16, y=1.02)

    variables = [
        ('abs_model_error', '(Absolute) Proficiency Model Error'),
        ('abs_conf_error', '(Absolute) Confidence Model Error'),
        ('abs_confScore_error', '(Absolute) Confidence Model Error (using scaled confidence)'),
        ('model_error', 'Proficiency Model Error [actual - prediction]'),
        ('conf_error', 'Confidence Model Error [actual - prediction]'),
        ('confScore_error', 'Confidence Model Error (using scaled confidence) [actual - prediction]')
    ]

    for level_idx, (df, level) in enumerate([
        (df_item, 'Item'),
        (df_item_agnostic, 'Item-Agnostic'),
        (df_skill, 'Skill')
    ]):
        base_row = level_idx * 2  # 0 for item, 2 for item-agnostic, 4 for skill
        
        for i, (var, title) in enumerate(variables):
            row = base_row + (i // 3)
            col = i % 3
            
            data = df[var].dropna()
            n_count = len(data)
            
            n, bins, patches = axs[row, col].hist(data, bins=30, density=True, alpha=0.6)


               # New code with try-except and variance check:
            try:
                # Check if data has enough variance for KDE
                if len(data) > 1 and np.var(data) > 1e-10:  # Adjust threshold as needed
         
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    axs[row, col].plot(x_range, kde(x_range), 'r-', lw=2)

                else:
                    # For low variance data, just add a note instead of KDE
                    logger.info(f"Skipping KDE for {level} {title} due to insufficient variance")
            except Exception as e:
                logger.info(f"Error in KDE calculation for {level} {title}: {str(e)}")

            
            mean = data.mean()
            std = data.std()
            
            axs[row, col].axvline(mean, color='red', linestyle='--', alpha=0.8, 
                                label=f'Mean: {mean:.3f}\nStd: {std:.3f}\nN: {n_count:,}')
            axs[row, col].axvline(mean + std, color='green', linestyle=':', alpha=0.5)
            axs[row, col].axvline(mean - std, color='green', linestyle=':', alpha=0.5)
            axs[row, col].axvline(0, color='black', linestyle='-', alpha=0.2)
            
            axs[row, col].set_ylim(bottom=0)
            axs[row, col].set_title(f'{level} {title}', fontsize=16)
            axs[row, col].set_xlabel(var, fontsize=14)
            axs[row, col].set_ylabel('Density', fontsize=14)
            axs[row, col].legend(fontsize=12)
            axs[row, col].tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig(f'error_distributions_{timestamp}.png')
    plt.close()

    # Combined response analysis for item, item-agnostic, and skill
    fig2, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig2.suptitle('Error Metrics by Number of Previous Responses - Item vs Item-Agnostic vs Skill Level', fontsize=16)

    colors = {'Item': 'blue', 'Item-Agnostic': 'green', 'Skill': 'red'}
    for df_idx, (df, level) in enumerate([
        (df_item, 'Item'),
        (df_item_agnostic, 'Item-Agnostic'),
        (df_skill, 'Skill')
    ]):
        response_metrics = []
        for n in range(11):
            mask = data_scaled['num_responses'] == n
            if mask.any():
                subset_prof_error = df['abs_model_error'][mask]
                subset_conf_error = df['abs_conf_error'][mask]
                response_metrics.append({
                    'num_responses': n,
                    'count': len(subset_prof_error),
                    'prof_mae': np.mean(subset_prof_error),
                    'prof_mse': np.mean(subset_prof_error ** 2),
                    'prof_rmse': np.sqrt(np.mean(subset_prof_error ** 2)),
                    'conf_mae': np.mean(subset_conf_error),
                    'conf_mse': np.mean(subset_conf_error ** 2),
                    'conf_rmse': np.sqrt(np.mean(subset_conf_error ** 2))
                })

        df_metrics = pd.DataFrame(response_metrics)
        color = colors[level]
        
        # Sample counts
        axs[0, 0].plot(df_metrics['num_responses'], df_metrics['count'], 
                      marker='o', linewidth=2, markersize=8, 
                      label=f'{level}', color=color)
        
        # MAE
        axs[0, 1].plot(df_metrics['num_responses'], df_metrics['prof_mae'], 
                      marker='o', linewidth=2, markersize=8,
                      label=f'{level} Proficiency', color=color)
        axs[0, 1].plot(df_metrics['num_responses'], df_metrics['conf_mae'], 
                      marker='s', linewidth=2, markersize=8,
                      label=f'{level} Confidence', color=color, linestyle='--')
        
        # MSE
        axs[1, 0].plot(df_metrics['num_responses'], df_metrics['prof_mse'], 
                      marker='o', linewidth=2, markersize=8,
                      label=f'{level} Proficiency', color=color)
        axs[1, 0].plot(df_metrics['num_responses'], df_metrics['conf_mse'], 
                      marker='s', linewidth=2, markersize=8,
                      label=f'{level} Confidence', color=color, linestyle='--')
        
        # RMSE
        axs[1, 1].plot(df_metrics['num_responses'], df_metrics['prof_rmse'], 
                      marker='o', linewidth=2, markersize=8,
                      label=f'{level} Proficiency', color=color)
        axs[1, 1].plot(df_metrics['num_responses'], df_metrics['conf_rmse'], 
                      marker='s', linewidth=2, markersize=8,
                      label=f'{level} Confidence', color=color, linestyle='--')

        # Log metrics
        logger.info(f"\n{level} Level - Metrics by number of previous responses:")
        for metrics in response_metrics:
            logger.info(f"\nResponses: {metrics['num_responses']}")
            logger.info(f"  Sample count: {metrics['count']}")
            logger.info(f"  Proficiency Model:")
            logger.info(f"    MAE: {metrics['prof_mae']:.5f}")
            logger.info(f"    MSE: {metrics['prof_mse']:.5f}")
            logger.info(f"    RMSE: {metrics['prof_rmse']:.5f}")
            logger.info(f"  Confidence Model:")
            logger.info(f"    MAE: {metrics['conf_mae']:.5f}")
            logger.info(f"    MSE: {metrics['conf_mse']:.5f}")
            logger.info(f"    RMSE: {metrics['conf_rmse']:.5f}")

    # Set titles and labels for the response analysis plots
    titles = ['Number of Samples by Previous Responses',
             'Mean Absolute Error by Previous Responses',
             'Mean Squared Error by Previous Responses',
             'Root Mean Squared Error by Previous Responses']
    ylabels = ['Number of Samples', 'MAE', 'MSE', 'RMSE']
    
    for idx, (ax, title, ylabel) in enumerate(zip(axs.flat, titles, ylabels)):
        ax.set_title(title)
        ax.set_xlabel('Number of Previous Responses')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'error_by_responses_{timestamp}.png')
    plt.close()
    
    
def batch_predict_all_questions_for_skills_w_conf(proficiency_model, confidence_model, confidence_score_scaler,transformed_loaded_ipd, X_input, skill_ids, feature_names, proficiency_scaler, valid_skill_ids, SigmoidTransformOutput=False, sigmoid_params=None):
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
    
    if proficiency_scaler:
        # Scale the data
        X_all_questions_scaled = proficiency_scaler.transform(all_predictions)
    else:
        X_all_questions_scaled = all_predictions.copy()
        

    # Make predictions for all questions at once
    dmatrix = xgb.DMatrix(X_all_questions_scaled)
    
    #ADD SIGMOID HERE 
    if SigmoidTransformOutput:
        def apply_sigmoid(predictions, params):
            a, b = params
            return 1 / (1 + np.exp(-a * (predictions - b)))  # b is the centering 
        predictions = apply_sigmoid(proficiency_model.predict(dmatrix), sigmoid_params)
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
    
    
def InferenceTesting(input_folder, proficiency_model_output_folder, confidence_model_output_folder, item_params_file,
                     CorrectnessBinary, TestMode=False, FUTURE_WINDOW=0, scalerFlag = False, 
                     raw_data_folder=None, test_ids=None, column_names=None,
                     memorySaver = True, UseSigmoidTransform = False, 
                     SigmoidTransformOutput = False, ItemAgnostic=False,
                     ItemAgnosticDoubleFit=False,
                     BATCH_SIZE=10000, smallInferenceRun=None):
    logger.info("Starting Inference Testing on reserved test set")

    custom_metric = reg_custom_metric
    
    if scalerFlag:
        # Load the proficiency model and scaler
        proficiency_scaler_filename = os.path.join(proficiency_model_output_folder, 'nan_preserving_scaler_proficiency.joblib')
        proficiency_scaler = joblib.load(proficiency_scaler_filename)
        logger.info(f"Loaded NanPreservingScaler from {proficiency_scaler_filename}")
    else:
        proficiency_scaler = None

    proficiency_model_file = os.path.join(proficiency_model_output_folder, 'xgb_model.json')
    #proficiency_model = xgb.Booster()
    #proficiency_model.load_model(proficiency_model_file)
    if UseSigmoidTransform:
        logger.info("Loading sigmoid-transformed proficiency model...")
        proficiency_model = SigmoidXGBRegressor()
        proficiency_model.load_model(proficiency_model_file)
    else:
        proficiency_model = xgb.Booster()
        proficiency_model.load_model(proficiency_model_file)    
    logger.info(f"Loaded proficiency model from {proficiency_model_file}")

    
    if SigmoidTransformOutput:
        # To load and use
        with open(os.path.join(proficiency_model_output_folder, 'sigmoid_params.json'), 'r') as f:
            params = json.load(f)
            sigmoid_params = [params['a'], params['b']] 
        logger.info(f"Loaded sigmoid transformer for proficiency model: sigmoid_params.json from {proficiency_model_output_folder}")
        logger.info(f"Sigmoid parameters, a,b: {params['a']},{params['b']}")

        def apply_sigmoid(predictions, params):
            a, b = params
            return 1 / (1 + np.exp(-a * (predictions - b)))  # b is the centering point
    else:
        sigmoid_params = None

        
    # Load the confidence model
    confidence_model_file = os.path.join(confidence_model_output_folder, 'confidence_xgb_model.json')
    confidence_model = xgb.Booster()
    confidence_model.load_model(confidence_model_file)
    logger.info(f"Loaded confidence model from {confidence_model_file}")
    
    #Load Confidence Score Scaler
    confidence_model_score_transformer_filename = os.path.join(confidence_model_output_folder, 
                                                               'percentile_rank_calculator.pkl')

    try:
        # Try to load the pkl file
        confidence_score_scaler = PercentileRankCalculator.load(confidence_model_score_transformer_filename)
        logger.info(f"Confidence model Score Transformer loaded from {confidence_model_score_transformer_filename}")
    except FileNotFoundError: #USE JSON IF NO PKL FOUND (OLD MODELS ONLY HAVE THE FINAL JSON SAVED)
        # If pkl file not found, try json instead
        json_filename = confidence_model_score_transformer_filename.replace('.pkl', '.json')
        with open(json_filename, 'r') as f:
            data = json.load(f)
        confidence_score_scaler = PercentileRankCalculator([])
        confidence_score_scaler.scores = np.array(data['scores'])
        confidence_score_scaler.ranks = np.array(data['ranks'])
        confidence_score_scaler.total_scores = data['total_scores']
        logger.info(f"Using JSON instead. Loaded from {json_filename}")
    
    #confidence_score_scaler = PercentileRankCalculator.load(confidence_model_score_transformer_filename)
    #logger.info(f"Confidence model Score Transformer loaded from {confidence_model_score_transformer_filename}")
    

    # Load feature names
    with open(os.path.join(proficiency_model_output_folder, 'feature_names.json'), 'r') as f:
        proficiency_features = json.load(f)
    with open(os.path.join(confidence_model_output_folder, 'confidence_feature_names.json'), 'r') as f:
        confidence_features = json.load(f)

    # Load and transform IPD data
    transformed_loaded_ipd = pd.read_csv(item_params_file)
    #transformed_loaded_ipd = item_params.copy()
    transformed_loaded_ipd['LOG_sample_size'] = np.log(transformed_loaded_ipd['sample_size'])
    transformed_loaded_ipd = transformed_loaded_ipd.drop('sample_size', axis=1)
    logger.info(f"Loaded IPD data {item_params_file}")

    # Get all parquet files in the input folder
    all_files = glob.glob(os.path.join(input_folder, "*.parquet"))

    #load all or only a small amount (for pipeline testing)
    if TestMode:
        logger.info("Test Mode: Finding files with sufficient data...")
        valid_files = []
        min_rows = 10  # Minimum rows needed for training

        for file in all_files:
            # Load and check data size
            df = pd.read_parquet(file)
            if len(df) >= min_rows:
                valid_files.append(file)
                logger.info(f"Found valid file: {file} with {len(df)} rows")
                if len(valid_files) == 2:  # We found enough files
                    break
        if len(valid_files) < 2:
            raise ValueError(f"Could not find 2 files with at least {min_rows} rows each. Found {len(valid_files)} valid files.")
        all_files = valid_files
        logger.info(f"Test mode will use these files: {valid_files}")
    else:
        if smallInferenceRun is not None:
            logger.info(f"Small Inference Run Mode: Finding {smallInferenceRun} files with sufficient data...")
            valid_files = []
            min_rows = 10  # Minimum rows needed for training

            for file in all_files:
                # Load and check data size
                df = pd.read_parquet(file)
                if len(df) >= min_rows:
                    valid_files.append(file)
                    logger.info(f"Found valid file: {file} with {len(df)} rows")
                    if len(valid_files) == smallInferenceRun:  # We found enough files
                        break
            if len(valid_files) < smallInferenceRun:
                raise ValueError(f"Could not find {smallInferenceRun} files with at least {min_rows} rows each. Found {len(valid_files)} valid files.")
            all_files = valid_files
            logger.info(f"Small inference run mode will use these files: {valid_files}")


    # Load threshold parameters, otherwise use default
    try:
        threshold_file = os.path.join(proficiency_model_output_folder, 'threshold_params.json')
        with open(threshold_file, 'r') as f:
            threshold_params = json.load(f)
        model_struggling_threshold = threshold_params['struggling_threshold']
        model_proficiency_threshold = threshold_params['proficiency_threshold']
        logger.info(f"Loaded threshold parameters: struggling={model_struggling_threshold:.3f}, proficiency={model_proficiency_threshold:.3f}")
    except:
        model_struggling_threshold=0.6
        model_proficiency_threshold=0.8
        logger.info(f"Using DEFAULT threshold parameters: struggling={model_struggling_threshold:.3f}, proficiency={model_proficiency_threshold:.3f}")

    # Final evaluation
    from collections import defaultdict
    logger.info("Final model metrics using reserved test set.")
    item_prof_metric_sums = defaultdict(float)
    item_conf_metric_sums = defaultdict(float)
    item_cs_metric_sums = defaultdict(float)
    skill_prof_metric_sums = defaultdict(float)
    skill_conf_metric_sums = defaultdict(float)
    skill_cs_metric_sums = defaultdict(float)
    # Need to add new dictionaries for item_agnostic metrics
    item_agnostic_prof_metric_sums = defaultdict(float)
    item_agnostic_conf_metric_sums = defaultdict(float)
    item_agnostic_cs_metric_sums = defaultdict(float)
    logger.info(f"Erasing next-item information for item-agnostic inference testing. This is used when we are forecasting more than one item into the future, or testing a non-individual-item probability calculation for skill.")
    erase_columns = ["discriminability", "difficulty", "guessing", "inattention", "discriminability_error",
                         "difficulty_error", "guessing_error", "inattention_error", "auc_roc", "optimal_threshold", "tpr",
                         "tnr", "skill_optimal_threshold", "student_mean_accuracy"]

    num_test_files = len(all_files)
    
    logger.info(f"Found {len(all_files)} files.")

    #for median of confidence model
    cm_targets = []
    cm_predictions = []

    #collect valid ids once
    valid_skill_ids = set(transformed_loaded_ipd['skill_id'].unique())
        
    for i, file in enumerate(all_files):
        logger.info(f"Test set {i+1}:")
        
        loaded_features = proficiency_features[:]
        #if future windowin,  need to load it too
        if FUTURE_WINDOW>0:
            loaded_features += [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]
        #add skill
        loaded_features +=['SKILL']
        #exclusion_list = ['STUDENTID', 'SKILL', 'QUESTIONID', 'CORRECTNESS', 'DURATIONSECONDS',
        #'ANSWERPOSITIONINSESSION', 'EVENT_RANK']
        
        # Load and preprocess data
        data = load_single_parquet_file(file, loaded_features, 'CORRECTNESS', CorrectnessBinary)

        if scalerFlag:
            # Scale the data
            data_scaled[proficiency_features] = proficiency_scaler.transform(data[proficiency_features])
        else:
            data_scaled = data

        #Make prediction 'item agnostic' (ignore next item information)
        #alternatives: previously, we used that information, even if we were looking at 'next 3/5'
        #alternative #2: precise skill prob calculation using all items in skill (used in inference wrapper),
        #        not implemented here yet
        
        data_scaled_agnostic = data_scaled.copy()
        data_scaled_agnostic[erase_columns] = np.nan
            
            
        #ITEM Proficiency model predictions;; dtrain has label from load_single_parquet_file
        dproficiency = xgb.DMatrix(data_scaled[proficiency_features], label=data_scaled['CORRECTNESS'])
        dproficiency_agnostic = xgb.DMatrix(data_scaled_agnostic[proficiency_features], label=data_scaled_agnostic['CORRECTNESS'])

        if SigmoidTransformOutput:
            item_proficiency_predictions = apply_sigmoid(proficiency_model.predict(dproficiency), sigmoid_params)
            item_agnostic_proficiency_predictions = apply_sigmoid(proficiency_model.predict(dproficiency_agnostic), sigmoid_params)
        else:
            item_proficiency_predictions = np.clip(proficiency_model.predict(dproficiency),0,1)
            item_agnostic_proficiency_predictions = np.clip(proficiency_model.predict(dproficiency_agnostic),0,1)

        #ITEM CONFIDENCE
        # Prepare data for confidence model: dtrain w/ features, label; + prediction
        #first append prediction as input
        data_scaled['proficiency_model_prediction'] = item_proficiency_predictions
        data_scaled_agnostic['proficiency_model_prediction'] = item_agnostic_proficiency_predictions
        #need to add label for dtrain
        #single trial 'error'; multi-trial 'error' if future windowing:
        if FUTURE_WINDOW == 0:
            data_scaled['error'] = data_scaled['CORRECTNESS'] - data_scaled['proficiency_model_prediction']
            data_scaled_agnostic['error'] = data_scaled_agnostic['CORRECTNESS'] - data_scaled_agnostic['proficiency_model_prediction']
        else:
            logger.info("Using future window of "+str(FUTURE_WINDOW)+" for confidence model inference metrics.")
            targets = ['CORRECTNESS'] + [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]
            data_scaled['error'] = abs(data_scaled[targets].mean(axis=1).values - data_scaled['proficiency_model_prediction'])
            data_scaled_agnostic['error'] = abs(data_scaled_agnostic[targets].mean(axis=1).values - data_scaled_agnostic['proficiency_model_prediction'])
            
            
        dconfidence = xgb.DMatrix(data_scaled[confidence_features], label=data_scaled['error'])
        item_confidence_predictions = np.clip(confidence_model.predict(dconfidence),0,1)

        dconfidence_agnostic = xgb.DMatrix(data_scaled_agnostic[confidence_features], label=data_scaled_agnostic['error'])
        item_agnostic_confidence_predictions = np.clip(confidence_model.predict(dconfidence_agnostic),0,1)
        
        
        #SKILL PROF & CONF
        skill_ids = data_scaled['SKILL'].tolist()

        #BEGIN OUTER BATCH LOOP HERE (For memory constraints)
        #predictions, confidence_predicted_error, confidence_scores, all_question_ids, # Initialize arrays for final results
        skill_proficiency_predictions = np.full(len(skill_ids), np.nan)
        skill_confidence_predictions = np.full(len(skill_ids), np.nan)
        skill_confidence_scores = np.full(len(skill_ids), np.nan)

        # Process data in batches
        for start_idx in range(0, len(data_scaled), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(data_scaled))

            # Get batch data
            batch_data = data_scaled.iloc[start_idx:end_idx]
            batch_skill_ids = skill_ids[start_idx:end_idx]

            # Process batch here --
            # using computationally faster batch prediction from modified infernece wrapper
            (batched_predictions, 
             batched_confidence_predicted_error,
             batched_skill_confidence_scores,
             question_ids,
             prediction_counts,
             valid_indices) = batch_predict_all_questions_for_skills_w_conf(
                proficiency_model,
                confidence_model,
                confidence_score_scaler,
                transformed_loaded_ipd,
                batch_data[proficiency_features],
                batch_skill_ids,
                proficiency_features,
                proficiency_scaler,
                valid_skill_ids,
                SigmoidTransformOutput, 
                sigmoid_params
            )
            
            # Process batch predictions if we have any
            if len(batched_predictions) > 0:
                valid_predictions = np.split(batched_predictions, np.cumsum(prediction_counts)[:-1])
                valid_uncertainties = np.split(batched_confidence_predicted_error, np.cumsum(prediction_counts)[:-1])
                valid_confidence_scores = np.split(batched_skill_confidence_scores, np.cumsum(prediction_counts)[:-1])

                for i, idx in enumerate(valid_indices):
                    if prediction_counts[idx] > 0 and len(valid_predictions[i]) > 0:
                        actual_idx = start_idx + idx  # Adjust index for the full dataset
                        skill_proficiency_predictions[actual_idx] = np.nanmean(valid_predictions[i])
                        skill_confidence_predictions[actual_idx] = np.nanmean(valid_uncertainties[i])
                        skill_confidence_scores[actual_idx] = np.nanmean(valid_confidence_scores[i])

            # Optional: Clear batch memory
            del batched_predictions, batched_confidence_predicted_error, batched_skill_confidence_scores

        # Debug print counts (unchanged from original)
        logger.info(f'skill_proficiency_predictions - NaN: {np.isnan(skill_proficiency_predictions).sum()}, non-NaN: {(~np.isnan(skill_proficiency_predictions)).sum()}')        
        
        #SCALED scores
        # Skill confidence scores included in batch process
        item_confidence_scores = confidence_score_scaler.get_percentile_rank(1- item_confidence_predictions)
        item_agnostic_confidence_scores = confidence_score_scaler.get_percentile_rank(1- item_agnostic_confidence_predictions)
#         skill_confidence_scores = confidence_score_scaler.get_percentile_rank(1 - skill_confidence)

        #just use the scaled score's 50%tile for the CScore
        #median search of confidnece model for imputed metrics (USES ITEM FOR NOW)
        cm_targets.append(np.median(data_scaled['error'].values))
        cm_predictions.append(np.median(item_confidence_predictions))
        median_threshold_tgt = np.mean(cm_targets)
        median_threshold_pred = np.mean(cm_predictions)

        logger.info(f"item_AWARE_prof_metrics")
        item_prof_metrics = custom_metric(item_proficiency_predictions, dproficiency, True, True)[0]  # Assuming custom_metric returns a list with one tuple
        logger.info(f"item_conf_metrics")
        item_conf_metrics = custom_metric(item_confidence_predictions, dconfidence, True, True, median_threshold_tgt, median_threshold_pred)[0]  
        logger.info(f"item_cs_metrics")
        item_cs_metrics = custom_metric(item_confidence_scores/100, dconfidence, True, True)[0]  
        
        logger.info(f"item_AGNOSTIC_prof_metrics")        
        item_agnostic_prof_metrics = custom_metric(item_agnostic_proficiency_predictions, dproficiency_agnostic, True, True)[0]
        item_agnostic_conf_metrics = custom_metric(item_agnostic_confidence_predictions, dconfidence_agnostic, True, True)[0]
        item_agnostic_cs_metrics = custom_metric(item_agnostic_confidence_scores/100, dconfidence_agnostic, True, True)[0]

        logger.info(f"skill_prof_metrics")
        skill_prof_metrics = custom_metric(skill_proficiency_predictions, dproficiency, True, True)[0]  # Assuming custom_metric returns a list with one tuple
        logger.info(f"skill_conf_metrics")
        skill_conf_metrics = custom_metric(skill_confidence_predictions, dconfidence, True, True, median_threshold_tgt, median_threshold_pred)[0]  
        logger.info(f"skill_cs_metrics")
        skill_cs_metrics = custom_metric(skill_confidence_scores/100, dconfidence, True, True)[0]  

        
    #aggregate metrics
    for j in range(0, len(item_prof_metrics), 2):
        item_prof_metric_name, item_prof_metric_value = item_prof_metrics[j], item_prof_metrics[j+1]
        item_prof_metric_sums[item_prof_metric_name] += item_prof_metric_value
    for j in range(0, len(item_conf_metrics), 2):
        item_conf_metric_name, item_conf_metric_value = item_conf_metrics[j], item_conf_metrics[j+1]
        item_conf_metric_sums[item_conf_metric_name] += item_conf_metric_value
    for j in range(0, len(item_cs_metrics), 2):
        item_cs_metric_name, item_cs_metric_value = item_cs_metrics[j], item_cs_metrics[j+1]
        item_cs_metric_sums[item_cs_metric_name] += item_cs_metric_value
        
    for j in range(0, len(item_agnostic_prof_metrics), 2):
        item_agnostic_prof_metric_name, item_agnostic_prof_metric_value = item_agnostic_prof_metrics[j], item_agnostic_prof_metrics[j+1]
        item_agnostic_prof_metric_sums[item_agnostic_prof_metric_name] += item_agnostic_prof_metric_value
    for j in range(0, len(item_agnostic_conf_metrics), 2):
        item_agnostic_conf_metric_name, item_agnostic_conf_metric_value = item_agnostic_conf_metrics[j], item_agnostic_conf_metrics[j+1]
        item_agnostic_conf_metric_sums[item_agnostic_conf_metric_name] += item_agnostic_conf_metric_value
    for j in range(0, len(item_agnostic_cs_metrics), 2):
        item_agnostic_cs_metric_name, item_agnostic_cs_metric_value = item_agnostic_cs_metrics[j], item_agnostic_cs_metrics[j+1]
        item_agnostic_cs_metric_sums[item_agnostic_cs_metric_name] += item_agnostic_cs_metric_value


    for j in range(0, len(skill_prof_metrics), 2):
        skill_prof_metric_name, skill_prof_metric_value = skill_prof_metrics[j], skill_prof_metrics[j+1]
        skill_prof_metric_sums[skill_prof_metric_name] += skill_prof_metric_value
    for j in range(0, len(skill_conf_metrics), 2):
        skill_conf_metric_name, skill_conf_metric_value = skill_conf_metrics[j], skill_conf_metrics[j+1]
        skill_conf_metric_sums[skill_conf_metric_name] += skill_conf_metric_value
    for j in range(0, len(skill_cs_metrics), 2):
        skill_cs_metric_name, skill_cs_metric_value = skill_cs_metrics[j], skill_cs_metrics[j+1]
        skill_cs_metric_sums[skill_cs_metric_name] += skill_cs_metric_value

    #End of file loop

    
    
    #Output: raw SQL, JSONified input, & featured engineered model input

    #Loads the raw SQL data (raw_data)
    if raw_data_folder:
        logger.info(f"Looking for raw files in: {raw_data_folder}")
        raw_files_list = glob.glob(raw_data_folder)
        logger.info(f"Found {len(raw_files_list)} raw files.")
        raw_files = {os.path.basename(f): f for f in raw_files_list}
        raw_file = raw_files.get(os.path.basename(file))
        if raw_file:
            logger.info(f"Found matching raw file: {raw_file}")
            raw_data = pd.read_parquet(raw_file)
            raw_data.columns = column_names
            # Filter by test_ids just like in feature engineering
            logger.info(f"Loaded raw data with {len(raw_data)} rows before filtering for test_ids.")
            raw_data_filtered = raw_data[raw_data['STUDENTID'].isin(test_ids)]
            raw_data_filtered = raw_data_filtered.reset_index(drop=True)
            logger.info(f"Loaded raw data with {len(raw_data_filtered)} rows after filtering for test_ids.")
            logger.info(f"Feature engineered data has {len(data_scaled)} rows.")
        else:
            logger.info(f"No matching raw file found for {os.path.basename(file)}")

    #reload the feature engineered data with all columns
    #load ALL feature names for the feature engineered files            
    with open(os.path.join(input_folder, "feature_engineered_column_names.json"), 'r') as jsonfilecols:
        all_fe_column_names = json.load(jsonfilecols)
    reloaded_fe_data = pd.read_parquet(file, columns=all_fe_column_names)
        
    # Calculate and log average metrics
    logger.info("Average metrics across all test sets (item aware/item agnostic/skill: proficiency & confidence models & confidence scores):")        
    logger.info("Average ITEM AWARE PROFICIENCY MODEL metrics across all test sets:")
    for item_prof_metric_name, item_prof_metric_sum in item_prof_metric_sums.items():
        item_prof_avg_metric = np.nanmean(item_prof_metric_sum)
        logger.info(f"{item_prof_metric_name}: {item_prof_avg_metric:.5f}")

    logger.info("Average ITEM AWARE CONFIDENCE MODEL metrics across all test sets:")
    for item_conf_metric_name, item_conf_metric_sum in item_conf_metric_sums.items():
        item_conf_avg_metric = np.nanmean(item_conf_metric_sum)
        logger.info(f"{item_conf_metric_name}: {item_conf_avg_metric:.5f}")

    logger.info("Average ITEM AWARE CONFIDENCE SCORE metrics across all test sets:")
    for item_cs_metric_name, item_cs_metric_sum in item_cs_metric_sums.items():
        item_cs_avg_metric = np.nanmean(item_cs_metric_sum)
        logger.info(f"{item_cs_metric_name}: {item_cs_avg_metric:.5f}")

    logger.info("Average ITEM AGNOSTIC PROFICIENCY MODEL metrics across all test sets:")
    for item_agnostic_prof_metric_name, item_agnostic_prof_metric_sum in item_agnostic_prof_metric_sums.items():
        item_agnostic_prof_avg_metric = np.nanmean(item_agnostic_prof_metric_sum)
        logger.info(f"{item_agnostic_prof_metric_name}: {item_agnostic_prof_avg_metric:.5f}")

    logger.info("Average ITEM AGNOSTIC CONFIDENCE MODEL metrics across all test sets:")
    for item_agnostic_conf_metric_name, item_agnostic_conf_metric_sum in item_agnostic_conf_metric_sums.items():
        item_agnostic_conf_avg_metric = np.nanmean(item_agnostic_conf_metric_sum)
        logger.info(f"{item_agnostic_conf_metric_name}: {item_agnostic_conf_avg_metric:.5f}")

    logger.info("Average ITEM AGNOSTIC CONFIDENCE SCORE metrics across all test sets:")
    for item_agnostic_cs_metric_name, item_agnostic_cs_metric_sum in item_agnostic_cs_metric_sums.items():
        item_agnostic_cs_avg_metric = np.nanmean(item_agnostic_cs_metric_sum)
        logger.info(f"{item_agnostic_cs_metric_name}: {item_agnostic_cs_avg_metric:.5f}")
        
    logger.info("Average SKILL PROFICIENCY MODEL metrics across all test sets:")
    for skill_prof_metric_name, skill_prof_metric_sum in skill_prof_metric_sums.items():
        skill_prof_avg_metric = np.nanmean(skill_prof_metric_sum)
        logger.info(f"{skill_prof_metric_name}: {skill_prof_avg_metric:.5f}")

    logger.info("Average SKILL CONFIDENCE MODEL metrics across all test sets:")
    for skill_conf_metric_name, skill_conf_metric_sum in skill_conf_metric_sums.items():
        skill_conf_avg_metric = np.nanmean(skill_conf_metric_sum)
        logger.info(f"{skill_conf_metric_name}: {skill_conf_avg_metric:.5f}")

    logger.info("Average SKILL CONFIDENCE SCORE metrics across all test sets:")
    for skill_cs_metric_name, skill_cs_metric_sum in skill_cs_metric_sums.items():
        skill_cs_avg_metric = np.nanmean(skill_cs_metric_sum)
        logger.info(f"{skill_cs_metric_name}: {skill_cs_avg_metric:.5f}") 

    # After making predictions, mirror live dashboard
    #add_error_analysis(data_scaled, proficiency_predictions, confidence_predictions, confidence_score_predictions, logger)
    add_error_analysis(data_scaled, item_proficiency_predictions, item_confidence_predictions, item_confidence_scores, item_agnostic_proficiency_predictions, item_agnostic_confidence_predictions, item_agnostic_confidence_scores,
skill_proficiency_predictions, skill_confidence_predictions, skill_confidence_scores, logger, FUTURE_WINDOW=FUTURE_WINDOW)
        
    # Get 5 random sample indices
    logger.info("\nPrinting 5 random samples from the last file:")
    random_indices = np.random.choice(len(data_scaled), 5, replace=False)
    
    # Sample data logging section
    for i, sample_idx in enumerate(random_indices):
        logger.info(f"\nStudent #{i+1}:")

        # Log raw SQL format
        if raw_data_folder and 'raw_data' in locals():
            raw_sample = raw_data_filtered.iloc[sample_idx]
            logger.info("Raw input (Snowflake SQL):")
            for col, val in raw_sample.items():
                logger.info(f"  {col}: {val}")

        # Log feature-engineered version
        logger.info(f"Feature-engineered inputs (internal):")
        logger.info(f"{data_scaled[proficiency_features].iloc[sample_idx].to_dict()}")

        if raw_data_folder and 'raw_data' in locals():
            # Log JSON formatted version for live testing
            #includeCurrent=True flag added to duplicate current item (bc live API does this)
            includeCurrent=True
            json_version = transform_row_to_json(raw_sample, includeCurrent=includeCurrent)
            if includeCurrent:
                logger.info("JSON formatted submission input (for live testing, with duplicated current inputs):")
            else:
                logger.info("JSON formatted submission input (for live testing, no current vals in history aggregation):")
            logger.info(json.dumps(json_version, indent=2, default=str))

        # Log true values
        logger.info(f"True value: {data_scaled['CORRECTNESS'].iloc[sample_idx]}")

        if FUTURE_WINDOW == 0:
            confidence_true = data_scaled['CORRECTNESS'].iloc[sample_idx]
            logger.info(f"Confidence true value: {confidence_true} (1 value)")
        else:
            targets = ['CORRECTNESS'] + [f'FUTURE_CORRECTNESS_LAG_{i}' for i in range(1, FUTURE_WINDOW+1)]
            confidence_true = data_scaled[targets].iloc[sample_idx].mean()
            values = data_scaled[targets].iloc[sample_idx]
            values_str = ", ".join([f"{targets[i]}: {values[i]:.3f}" for i in range(len(targets))])
            logger.info(f"Confidence true value: {confidence_true:.3f} (average of {FUTURE_WINDOW + 1} values: {values_str})")
            
        logger.info(f"Item Proficiency model prediction: {item_proficiency_predictions[sample_idx]}")
        logger.info(f"Item Confidence model prediction: {item_confidence_predictions[sample_idx]}")     
        logger.info(f"Item Confidence Score Transformer (percentile ranker) prediction: {item_confidence_scores[sample_idx]}")     
        logger.info(f"Item AGNOSTIC Proficiency model prediction: {item_agnostic_proficiency_predictions[sample_idx]}")
        logger.info(f"Item AGNOSTIC Confidence model prediction: {item_agnostic_confidence_predictions[sample_idx]}")     
        logger.info(f"Item AGNOSTIC Confidence Score Transformer (percentile ranker) prediction: {item_agnostic_confidence_scores[sample_idx]}")     
        logger.info(f"Skill Proficiency model prediction: {skill_proficiency_predictions[sample_idx]}")
        logger.info(f"Skill Confidence model prediction: {skill_confidence_predictions[sample_idx]}")     
        logger.info(f"Skill Confidence Score Transformer (percentile ranker) prediction: {skill_confidence_scores[sample_idx]}")     


    # Default thresholds
    default_struggling_threshold = 0.6
    default_proficiency_threshold = 0.8

    for analysis_type, predictions in [('Skill', skill_proficiency_predictions), ('Item', item_proficiency_predictions),
    ('Item Agnostic', item_agnostic_proficiency_predictions)]:
        # First analysis with custom thresholds
        logger.info(f"\n{analysis_type} Proficiency Category Analysis (Custom Thresholds):")

        low_mask = predictions < model_struggling_threshold
        mid_mask = (predictions >= model_struggling_threshold) & (predictions < model_proficiency_threshold)
        high_mask = predictions >= model_proficiency_threshold

        total_predictions = len(predictions)
        low_count = np.sum(low_mask)
        mid_count = np.sum(mid_mask)
        high_count = np.sum(high_mask)

        low_percent = (low_count / total_predictions) * 100
        mid_percent = (mid_count / total_predictions) * 100
        high_percent = (high_count / total_predictions) * 100

        low_mean = np.mean(data_scaled['CORRECTNESS'][low_mask]) if low_count > 0 else 0
        mid_mean = np.mean(data_scaled['CORRECTNESS'][mid_mask]) if mid_count > 0 else 0
        high_mean = np.mean(data_scaled['CORRECTNESS'][high_mask]) if high_count > 0 else 0

        logger.info(f"\nStruggling Category (< {model_struggling_threshold}):")
        logger.info(f"Count: {low_count} ({low_percent:.1f}%)")
        logger.info(f"Mean Correctness: {low_mean:.3f}")
        logger.info(f"\nPracticing Category (>= {model_struggling_threshold} and < {model_proficiency_threshold}):")
        logger.info(f"Count: {mid_count} ({mid_percent:.1f}%)")
        logger.info(f"Mean Correctness: {mid_mean:.3f}")
        logger.info(f"\nProficient Category (>= {model_proficiency_threshold}):")
        logger.info(f"Count: {high_count} ({high_percent:.1f}%)")
        logger.info(f"Mean Correctness: {high_mean:.3f}")

        # Second analysis with default thresholds
        logger.info(f"\n{analysis_type} Proficiency Category Analysis (Default Thresholds):")

        low_mask = predictions < default_struggling_threshold
        mid_mask = (predictions >= default_struggling_threshold) & (predictions < default_proficiency_threshold)
        high_mask = predictions >= default_proficiency_threshold

        total_predictions = len(predictions)
        low_count = np.sum(low_mask)
        mid_count = np.sum(mid_mask)
        high_count = np.sum(high_mask)

        low_percent = (low_count / total_predictions) * 100
        mid_percent = (mid_count / total_predictions) * 100
        high_percent = (high_count / total_predictions) * 100

        low_mean = np.mean(data_scaled['CORRECTNESS'][low_mask]) if low_count > 0 else 0
        mid_mean = np.mean(data_scaled['CORRECTNESS'][mid_mask]) if mid_count > 0 else 0
        high_mean = np.mean(data_scaled['CORRECTNESS'][high_mask]) if high_count > 0 else 0

        logger.info(f"\nStruggling Category (< {default_struggling_threshold}):")
        logger.info(f"Count: {low_count} ({low_percent:.1f}%)")
        logger.info(f"Mean Correctness: {low_mean:.3f}")
        logger.info(f"\nPracticing Category (>= {default_struggling_threshold} and < {default_proficiency_threshold}):")
        logger.info(f"Count: {mid_count} ({mid_percent:.1f}%)")
        logger.info(f"Mean Correctness: {mid_mean:.3f}")
        logger.info(f"\nProficient Category (>= {default_proficiency_threshold}):")
        logger.info(f"Count: {high_count} ({high_percent:.1f}%)")
        logger.info(f"Mean Correctness: {high_mean:.3f}")
    
    # Calculate and log average metrics
    logger.info("Average metrics across all test sets (item aware/item agnostic/skill: proficiency & confidence models & confidence scores):")        
    logger.info("Average ITEM AWARE PROFICIENCY MODEL metrics across all test sets:")
    for item_prof_metric_name, item_prof_metric_sum in item_prof_metric_sums.items():
        item_prof_avg_metric = np.nanmean(item_prof_metric_sum)
        logger.info(f"{item_prof_metric_name}: {item_prof_avg_metric:.5f}")

    logger.info("Average ITEM AWARE CONFIDENCE MODEL metrics across all test sets:")
    for item_conf_metric_name, item_conf_metric_sum in item_conf_metric_sums.items():
        item_conf_avg_metric = np.nanmean(item_conf_metric_sum)
        logger.info(f"{item_conf_metric_name}: {item_conf_avg_metric:.5f}")

    logger.info("Average ITEM AWARE CONFIDENCE SCORE metrics across all test sets:")
    for item_cs_metric_name, item_cs_metric_sum in item_cs_metric_sums.items():
        item_cs_avg_metric = np.nanmean(item_cs_metric_sum)
        logger.info(f"{item_cs_metric_name}: {item_cs_avg_metric:.5f}")

    logger.info("Average ITEM AGNOSTIC PROFICIENCY MODEL metrics across all test sets:")
    for item_agnostic_prof_metric_name, item_agnostic_prof_metric_sum in item_agnostic_prof_metric_sums.items():
        item_agnostic_prof_avg_metric = np.nanmean(item_agnostic_prof_metric_sum)
        logger.info(f"{item_agnostic_prof_metric_name}: {item_agnostic_prof_avg_metric:.5f}")

    logger.info("Average ITEM AGNOSTIC CONFIDENCE MODEL metrics across all test sets:")
    for item_agnostic_conf_metric_name, item_agnostic_conf_metric_sum in item_agnostic_conf_metric_sums.items():
        item_agnostic_conf_avg_metric = np.nanmean(item_agnostic_conf_metric_sum)
        logger.info(f"{item_agnostic_conf_metric_name}: {item_agnostic_conf_avg_metric:.5f}")

    logger.info("Average ITEM AGNOSTIC CONFIDENCE SCORE metrics across all test sets:")
    for item_agnostic_cs_metric_name, item_agnostic_cs_metric_sum in item_agnostic_cs_metric_sums.items():
        item_agnostic_cs_avg_metric = np.nanmean(item_agnostic_cs_metric_sum)
        logger.info(f"{item_agnostic_cs_metric_name}: {item_agnostic_cs_avg_metric:.5f}")
        
    logger.info("Average SKILL PROFICIENCY MODEL metrics across all test sets:")
    for skill_prof_metric_name, skill_prof_metric_sum in skill_prof_metric_sums.items():
        skill_prof_avg_metric = np.nanmean(skill_prof_metric_sum)
        logger.info(f"{skill_prof_metric_name}: {skill_prof_avg_metric:.5f}")

    logger.info("Average SKILL CONFIDENCE MODEL metrics across all test sets:")
    for skill_conf_metric_name, skill_conf_metric_sum in skill_conf_metric_sums.items():
        skill_conf_avg_metric = np.nanmean(skill_conf_metric_sum)
        logger.info(f"{skill_conf_metric_name}: {skill_conf_avg_metric:.5f}")

    logger.info("Average SKILL CONFIDENCE SCORE metrics across all test sets:")
    for skill_cs_metric_name, skill_cs_metric_sum in skill_cs_metric_sums.items():
        skill_cs_avg_metric = np.nanmean(skill_cs_metric_sum)
        logger.info(f"{skill_cs_metric_name}: {skill_cs_avg_metric:.5f}") 


    logger.info("Inference Testing completed")
    
    
    
    
    
    
def run_full_pipeline(pipeline_id='_v0',                    # Was version_name
                    feature_set_id='_standardFeatures',     # Was feature_modification
                    model_type='_standard',                 # Was model_modification
                    doSnowflakeETL=False,
                    skillExplode=True,
                    CorrectnessBinary=False,
                    UseSigmoidTransform = False,     #trains model with sigmoid
                    SigmoidTransformOutput = False,  #uses untrained sigmoid on predictions, no training
                    SigmoidTransformOutputFit=False, #fit the sigmoid transformer
                    ItemAgnostic=False,              #Ignore next item information for Inference testing
                    ItemAgnosticFit=False,              #Ignore next item information for model fitting 
                    ItemAgnosticDoubleFit=False,
                    F1ObjectiveMeasure=False,
                    pullNewData = False,
                    query_file = 'querytransformedtable.txt',
                    raw_data_folder_path = 'StudentProficiencyData_ELA_09252024',
                    GenerateTestTrainSplit = False,
                    ItemParameterCalculate = False,
                    item_params_file = 'ELA_ItemParameters_3months.csv',
                    FeatureEngineering=False,
                    ProficiencyModelFit=False,
                    ConfidenceModelFit=False,
                    InferenceModeOn=False,
                    InferenceModeFeatureEngineering = False,
                    TestMode=False,
                    n_estimators=200,
                    FUTURE_WINDOW=0,
                    scalerFlag = False,
                    early_stopping_rounds=10,
                    INFERENCE_BATCH_SIZE=10000, 
                    smallInferenceRun=None,
                    log_folder='ML_pipeline_logs'):
    

    # Set up logging
    import os
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f'ML_training_pipeline_log_{pipeline_id}_{feature_set_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    setup_logger(log_file)
    logger.info(f"Starting model training. Logs will be saved to {log_file}")
    
    #Flag check
    # Log all parameters
    logger.info("Pipeline Configuration:")
    logger.info(f"pipeline_id: {pipeline_id}")
    logger.info(f"feature_set_id: {feature_set_id}")
    logger.info(f"model_type: {model_type}")
    
    logger.info(f"doSnowflakeETL: {doSnowflakeETL}")
    logger.info(f"skillExplode: {skillExplode}")
    logger.info(f"CorrectnessBinary: {CorrectnessBinary}")
    logger.info(f"ItemAgnostic: {ItemAgnostic}")
    logger.info(f"ItemAgnosticFit: {ItemAgnosticFit}")
    logger.info(f"ItemAgnosticDoubleFit: {ItemAgnosticDoubleFit}")
    logger.info(f"F1ObjectiveMeasure: {F1ObjectiveMeasure}")
    
    logger.info(f"SigmoidTransformOutputFit: {SigmoidTransformOutputFit}")
    logger.info(f"UseSigmoidTransform: {UseSigmoidTransform}")
    logger.info(f"SigmoidTransformOutput: {SigmoidTransformOutput}")     
    logger.info(f"pullNewData: {pullNewData}")
    logger.info(f"query_file: {query_file}")
    logger.info(f"raw_data_folder_path: {raw_data_folder_path}")
    logger.info(f"GenerateTestTrainSplit: {GenerateTestTrainSplit}")
    logger.info(f"ItemParameterCalculate: {ItemParameterCalculate}")
    logger.info(f"item_params_file: {item_params_file}")
    logger.info(f"FeatureEngineering: {FeatureEngineering}")
    logger.info(f"ProficiencyModelFit: {ProficiencyModelFit}")
    logger.info(f"ConfidenceModelFit: {ConfidenceModelFit}")
    logger.info(f"InferenceModeOn: {InferenceModeOn}")
    logger.info(f"InferenceModeFeatureEngineering: {InferenceModeFeatureEngineering}")
    logger.info(f"TestMode: {TestMode}")
    logger.info(f"n_estimators: {n_estimators}")
    logger.info(f"FUTURE_WINDOW: {FUTURE_WINDOW}")
    logger.info(f"scalerFlag: {scalerFlag}")
    logger.info(f"log_folder: {log_folder}")

   # File location definitions
    feature_engineering_output_folder = 'feature_engineered_train_parquets' +'_' +  feature_set_id + '/'
    proficiency_input_folder = feature_engineering_output_folder
    confidence_input_folder = feature_engineering_output_folder
    # Path to the model, scaler, & feature names
    proficiency_model_output_folder = 'proficiency_model_' + pipeline_id + '_' + feature_set_id + '_' + model_type
    confidence_model_output_folder = 'confidence_model_' + pipeline_id + '_' + feature_set_id + '_' + model_type
    proficiency_model_file = proficiency_model_output_folder + '/xgb_model.json'    
    test_feature_engineering_output_folder='inferenceTest_'+feature_engineering_output_folder
    train_ids_file = 'train_studentids_'+pipeline_id+'.parquet'
    skill_train_ids_file = 'skill_train_studentids_'+pipeline_id+'.parquet'
    item_train_ids_file = 'item_train_studentids_'+pipeline_id+'.parquet'
    test_ids_file='test_studentids_'+pipeline_id+'.parquet'
    param_ids_file='parameter_studentids_'+pipeline_id+'.parquet'


    
    #file location defintions
    logger.info("Folder Locations for input data and output artifacts:")
    logger.info(f'feature_engineering_output_folder: {feature_engineering_output_folder}')
    logger.info(f'proficiency_input_folder: {proficiency_input_folder}')
    logger.info(f'proficiency_model_output_folder: {proficiency_model_output_folder}')
    logger.info(f'confidence_input_folder: {confidence_input_folder}')
    logger.info(f'confidence_model_output_folder: {confidence_model_output_folder}')
    logger.info(f'proficiency_model_file : {proficiency_model_file}')
    

    #Query Snowflake; Convert to parquet 
    #(could change query fetch data format directly)
    import time
    import os
    import glob
    import SnowflakeETL
    
    starttime=time.time()
    
    if doSnowflakeETL:
        starttime=time.time()
        if skillExplode:
            query_files = [
                "SnowflakeETL_query1.txt",
                "SnowflakeETL_query2.txt",
                "SnowflakeETL_query3.txt"
            ]
        else:
            query_files = [
                "SnowflakeETL_query1.txt",
                "SnowflakeETL_query2-noExplode.txt",
                "SnowflakeETL_query3.txt"
            ]

        queries = []
        for file in query_files:
            with open(file, "r") as f:
                query = f.read()
                queries.append(query)
                logger.info(f'grabbing query from: {file}')
                logger.info(f'submitting query: {query}')

        SnowflakeETL.run_snowflake_queries_sequentially(queries)
        logger.info(f'ETL finished. {str(time.time()-starttime)} seconds.')

        
    if pullNewData: #True to import latest data, False to use existing data pull
        logger.info('Query to fetch from snowflake')
        #grab all SP archived data
        SnowflakeETL.grabAllDataFromSnowflake(stage='[REDACTED]',
                                 query_file = query_file,
                                 datastore=raw_data_folder_path,
                                 csvTrigger=False)
        logger.info(f'Data pull finished. Took {time.time()-starttime} seconds.')

    # Load column names
    with open('column_names.txt', 'r') as f:
        column_names = f.read().splitlines()
    logger.info(f'column_names.txt (from raw query): {column_names}')

    # Use glob to get a list of all parquet files in the raw data folder
    file_pattern = os.path.join(raw_data_folder_path, '*.parquet')
    file_list = glob.glob(file_pattern)
    logger.info(f"{len(file_list)} files found.")


    if GenerateTestTrainSplit:
        logger.info(f'Starting Test Train Param student ID split.')
        
        # Read unique studentids from all Parquet files in a directory
        unique_studentids = read_unique_studentids_from_parquet_files(file_pattern, column_names)

        logger.info(f"Total unique studentids: {len(unique_studentids)}")

        from sklearn.model_selection import train_test_split
        # Perform train-test split on unique studentids
        #train_ids, test_ids = train_test_split(unique_studentids, test_size=0.2, random_state=42)
        #parameter_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
        train_ids, test_ids = train_test_split(unique_studentids, test_size=0.1, random_state=42)
        parameter_ids, train_ids = train_test_split(train_ids, test_size=0.5, random_state=42)
        skill_train_ids, item_train_ids = train_test_split(train_ids, test_size=0.5, random_state=42)

        logger.info(f"Training set size: {len(train_ids)}")
        logger.info(f"Item Training set size: {len(item_train_ids)}")
        logger.info(f"Skill Training set size: {len(skill_train_ids)}")
        logger.info(f"Testing set size: {len(test_ids)}")
        logger.info(f"Parameter set size: {len(parameter_ids)}")

        # Save split studentids
        
        train_ids.to_frame().to_parquet(train_ids_file)
        test_ids.to_frame().to_parquet(test_ids_file)
        parameter_ids.to_frame().to_parquet(param_ids_file)
        item_train_ids.to_frame().to_parquet(item_train_ids_file)
        skill_train_ids.to_frame().to_parquet(skill_train_ids_file)

        logger.info(f'Test-train-parameter split finished.') 
        logger.info(f'  Train IDs file: {train_ids_file}')
        logger.info(f'  Item Train IDs file: {skill_train_ids_file}')
        logger.info(f'  Skill Train IDs file: {item_train_ids_file}')
        logger.info(f'  Test IDs file: {test_ids_file}')
        logger.info(f'  Parameter IDs file: {param_ids_file}')


    # Item Parameter calculation
    # Example usage:
    if ItemParameterCalculate:
        logger.info(f'Starting Item Param Calc.')

        # Construct the output filename

        #validation ids only
        parameter_ids = pd.read_parquet(param_ids_file)['STUDENTID'].tolist()
        logger.info(f'Imported {len(parameter_ids)} studentids for IP Calc.')
        
        
        process_skills_irtsdt(
            file_pattern=file_pattern,
            column_names=column_names,
            output_filename=item_params_file,
            train_ids=parameter_ids,
            version='3.0',
            binary=CorrectnessBinary,
            restart_index=0,
            TestMode=TestMode,
            DATA_LIMIT=500000)
        
    train_ids = pd.read_parquet(train_ids_file)['STUDENTID'].tolist()
    logger.info(f'Imported {len(train_ids)} studentids for train split.')
        
    # Featuring engineering on imported SQL data
    if FeatureEngineering:
        logger.info(f'Starting Feature Engineering.')
        
        import os
        if not os.path.exists(feature_engineering_output_folder):
                os.makedirs(feature_engineering_output_folder)

        import time
        starttime= time.time()
        process_parquet_files(input_pattern=file_pattern, 
                              output_folder=feature_engineering_output_folder, 
                              train_ids=train_ids, item_params_file=item_params_file, 
                              column_names=column_names, TestMode=TestMode, 
                              FUTURE_WINDOW=FUTURE_WINDOW)
        logger.info(f'Feature engineering done (for train set). took {time.time()-starttime} seconds.')


    #clear counter for reruns
    try:
        del reg_custom_metric
        del binary_custom_metric
    except:
        pass

    if ProficiencyModelFit:
        logger.info(f'Starting proficiency model training.')
        train_proficiency_model(input_folder=proficiency_input_folder,
                                model_output_folder=proficiency_model_output_folder,
                                CorrectnessBinary=CorrectnessBinary,
                                TestMode=TestMode, n_estimators=n_estimators, 
                                scalerFlag = scalerFlag,
                                early_stopping_rounds=early_stopping_rounds,
                                UseSigmoidTransform = UseSigmoidTransform,
                                SigmoidTransformOutput=SigmoidTransformOutput,
                                F1ObjectiveMeasure=F1ObjectiveMeasure,
                                ItemAgnosticFit=ItemAgnosticFit,
                                ItemAgnosticDoubleFit=ItemAgnosticDoubleFit)


    if SigmoidTransformOutputFit:
        logger.info(f'Starting sigmoid transform training.')
        train_skill_model(input_folder=proficiency_input_folder, 
                              model_output_folder=proficiency_model_output_folder,
                              proficiency_model_file=proficiency_model_file, 
                              CorrectnessBinary=CorrectnessBinary, 
                              TestMode=TestMode, 
                              FUTURE_WINDOW=FUTURE_WINDOW, 
                              UseSigmoidTransform = UseSigmoidTransform,
                              scalerFlag=scalerFlag,
                              ItemAgnosticFit=ItemAgnosticFit)

    if ConfidenceModelFit:
        logger.info(f'Starting confidence model training.')
        train_confidence_model(input_folder=confidence_input_folder,
                               model_output_folder=confidence_model_output_folder,
                               proficiency_model_file=proficiency_model_file,
                               CorrectnessBinary=CorrectnessBinary, 
                               FUTURE_WINDOW=FUTURE_WINDOW, 
                               TestMode=TestMode, n_estimators=n_estimators,
                               scalerFlag = scalerFlag, 
                               early_stopping_rounds=early_stopping_rounds,
                               UseSigmoidTransform = UseSigmoidTransform,
                               SigmoidTransformOutput=SigmoidTransformOutput,
                               ItemAgnosticFit=ItemAgnosticFit)

    if InferenceModeOn:
        logger.info(f'Starting inference mode testing.')
        test_ids = pd.read_parquet(test_ids_file)['STUDENTID'].tolist()
        logger.info(f'Imported {len(test_ids)} studentids for test split.')

        
        # Featuring engineering on imported SQL data
        if InferenceModeFeatureEngineering:
            
            import os
            if not os.path.exists(test_feature_engineering_output_folder):
                    os.makedirs(test_feature_engineering_output_folder)

            import time
            starttime= time.time()

            process_parquet_files(input_pattern=file_pattern, 
                      output_folder=test_feature_engineering_output_folder, 
                      train_ids=test_ids, item_params_file=item_params_file, 
                      column_names=column_names, TestMode=TestMode, 
                      FUTURE_WINDOW=FUTURE_WINDOW)
            logger.info(f'Feature engineering done (for test set). took {time.time()-starttime} seconds.')            

        InferenceTesting(test_feature_engineering_output_folder, proficiency_model_output_folder, 
                         confidence_model_output_folder, item_params_file,
                         CorrectnessBinary, TestMode=TestMode, FUTURE_WINDOW=FUTURE_WINDOW, 
                         scalerFlag = scalerFlag, raw_data_folder=file_pattern, 
                         test_ids=test_ids, column_names=column_names,
                         UseSigmoidTransform = UseSigmoidTransform,
                         SigmoidTransformOutput=SigmoidTransformOutput,
                         ItemAgnostic=ItemAgnostic, 
                         ItemAgnosticDoubleFit=ItemAgnosticDoubleFit,
                         BATCH_SIZE=INFERENCE_BATCH_SIZE, smallInferenceRun=smallInferenceRun)
    
       
    logger.info(f'Pipeline finished.')            
       
