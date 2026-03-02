# Educational Assessment ML System - Detailed Technical Documentation

This document provides comprehensive technical documentation for the four core components of [REDACTED]'s educational assessment ML system: ModelImplementationWSDK, ItemParametersCalculate, ProficiencyModelTrainingPipeline, and Snowflake-based ETL, with WithinSkill_ELA (as an example).

## Table of Contents

1. [ModelImplementationWSDK](#modelimplementationwsdk)
2. [ItemParametersCalculate](#itemparameterscalculate)  
3. [ProficiencyModelTrainingPipeline](#proficiencymodeltrainingpipeline)
4. [WithinSkill_ELA](#withinskill_ela)
5. [SnowflakeETL](#snowflakeetl)

---

## ModelImplementationWSDK

### Overview
**ModelImplementationWSDK** is a production deployment framework for AWS SageMaker machine learning models. It provides end-to-end automation for containerizing models, building Docker images, managing ECR repositories, deploying SageMaker endpoints, and maintaining production infrastructure with advanced features like auto-scaling, health monitoring, rolling updates, and automatic rollback capabilities.

### Core Architecture

The SDK is structured around project-specific deployment directories, each containing a complete deployment ecosystem:

```
ModelImplementationWSDK/
├── tools/                                    # Shared deployment utilities
│   ├── sagemaker_deploy_tools.py           # Core deployment functions
│   ├── BuildDockerAndPushToECR.ipynb       # Docker build utilities  
│   ├── DeployModelFromECR.ipynb            # Deployment orchestration
│   ├── TestCurrentEndpoint.ipynb           # Health testing tools
│   └── [monitoring and testing tools]      # Additional utilities
├── ELA_student_proficiency_model/           # Project-specific deployment
│   ├── container/                           # Docker container setup
│   │   ├── Dockerfile                       # Container definition
│   │   ├── inference.py                     # Flask inference application
│   │   ├── build_and_push.sh              # Build automation script
│   │   └── [model artifacts]               # Models, scalers, parameters
│   ├── BuildAndDeployDockerImageToSagemakerEndpoint.ipynb  # Main deployment script
│   └── RollbackEndpoint.ipynb             # Rollback utilities
└── custom_model_template/                   # Template for new projects
```

### Core Components

#### 1. Main Deployment Script (`BuildAndDeployDockerImageToSagemakerEndpoint.ipynb`)

The **2034-line comprehensive deployment orchestrator** that handles the complete end-to-end deployment pipeline:

**Configuration Management:**
```python
# Project Configuration
ecr_repository = "[REDACTED]-ml/ela-proficiency-model"
endpoint_name = '[REDACTED]'
instance_type = 'ml.c5.large'
account_id = '[REDACTED]'
region = 'us-east-1'

# Auto-scaling Configuration
min_capacity = 1
max_capacity = 100
target_value = 15  # Target invocations per instance per minute
scale_in_cooldown = 60   # seconds
scale_out_cooldown = 300 # seconds
```

**Step 1: Docker Build and ECR Push**
```python
# Automated Docker operations
def run_command(command):
    # Robust command execution with error handling
    
# ECR Authentication
run_command(f"aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.{REGION}.amazonaws.com")

# Repository Management
run_command(f"aws ecr describe-repositories --repository-names {ECR_REPOSITORY} || aws ecr create-repository --repository-name {ECR_REPOSITORY}")

# Image Build and Push
run_command(f"docker build -t {ECR_REPOSITORY}:{IMAGE_TAG} .")
run_command(f"docker tag {ECR_REPOSITORY}:{IMAGE_TAG} {AWS_ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPOSITORY}:{IMAGE_TAG}")
run_command(f"docker push {AWS_ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPOSITORY}:{IMAGE_TAG}")
```

**Step 2: Local Container Testing**
```python
def test_docker_inference(repository, tag, test_message):
    """
    Critical pre-deployment validation:
    1. Starts container locally on port 8080
    2. Sends real inference requests to /invocations endpoint
    3. Validates response format and predictions
    4. Ensures container stability before cloud deployment
    """
    try:
        run_command(f"docker run -d -p 8080:8080 --name test-container {repository}:{tag}")
        time.sleep(5)  # Container startup time
        
        response = requests.post('http://localhost:8080/invocations',
                               json=test_message,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            print("Response:", response.json())
            return True
        else:
            raise Exception(f"Test failed with status {response.status_code}")
    finally:
        # Cleanup
        run_command("docker stop test-container")
        run_command("docker rm test-container")
```

**Step 3: SageMaker Deployment with Rolling Updates**
```python
if not endpoint_exists(endpoint_name):
    # New endpoint creation
    model.deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name=endpoint_name)
    setup_auto_scaling(endpoint_name, min_capacity, max_capacity, target_value, scale_in_cooldown, scale_out_cooldown)
else:
    # Rolling update with scaling management
    update_endpoint_with_scaling(endpoint_name, image_uri, test_message, 
                                min_capacity, max_capacity, target_value, 
                                scale_in_cooldown, scale_out_cooldown)
```

**Step 4: Comprehensive Testing Suite**

The script includes extensive unit tests covering:

- **Basic Inference Testing**: Single prediction validation
- **Malformed Input Testing**: Double JSON serialization error handling  
- **Batch Processing**: Multiple simultaneous predictions
- **Edge Case Handling**: Empty history, missing duration data
- **Invalid Skill Testing**: Graceful degradation for unknown skills
- **Real Production Data**: Testing with actual student response patterns

```python
# Example test cases from the actual script
test_message = {
    "skillId": "ubuipdtp",
    "questionId": "question_478", 
    "eventTime": "2024-10-03T06:20:45.377160",
    "questionIdsHistory": ["question_442", "question_801", "question_633", "question_616"],
    "correctnessHistory": [100, 100, 100, 0, 0, 0, 100, 0, 0],
    "durationSecondsHistory": [75, 217, 54, 241, 97, 113, 58, 32, 54],
    "eventTimesHistory": [...]
}

# Batch testing with multiple skills including fake skill IDs
test_message_multiple_events = [...] # 5 simultaneous predictions

# Edge case testing
empty_test_message = {"skillId": "ubuipdtp", "questionId": "question_478", 
                     "eventTime": "", "questionIdsHistory": [...],
                     "correctnessHistory": [], "durationSecondsHistory": [], 
                     "eventTimesHistory": []}
```

#### 2. Advanced Deployment Tools (`tools/sagemaker_deploy_tools.py`)

**Sophisticated production deployment functions with enterprise-grade features:**

**Auto-Scaling Management:**
```python
def setup_auto_scaling(endpoint_name, min_capacity, max_capacity, target_value, 
                      scale_in_cooldown=300, scale_out_cooldown=300):
    """
    Comprehensive auto-scaling setup:
    1. Deletes existing scaling policies to prevent conflicts
    2. Deregisters existing scalable targets
    3. Registers new scalable target with capacity bounds
    4. Applies target tracking scaling policy based on invocations per instance
    5. Configures cooldown periods for scaling stability
    """
    
def deregister_scalable_target(endpoint_name):
    """Safely removes auto-scaling before endpoint updates"""
    
def register_scalable_target(endpoint_name, min_capacity, max_capacity):
    """Establishes scaling infrastructure for endpoints"""
```

**Rolling Updates with Rollback:**
```python
def update_endpoint(endpoint_name, new_image_uri, test_message, new_instance_type=None):
    """
    Production-safe endpoint updates:
    1. Captures current endpoint configuration for rollback
    2. Creates new model with timestamped naming convention
    3. Creates new endpoint configuration
    4. Performs rolling update to new configuration
    5. Runs health checks on updated endpoint
    6. Automatically rolls back if health checks fail
    """
    
def update_endpoint_with_scaling(endpoint_name, new_image_uri, test_message, 
                                min_capacity, max_capacity, target_value, 
                                scale_in_cooldown, scale_out_cooldown):
    """
    Complete update workflow:
    1. Validates endpoint is InService before starting
    2. Deregisters auto-scaling to prevent conflicts during update
    3. Performs endpoint update with health validation
    4. Re-establishes auto-scaling configuration
    """
```

**Health Monitoring:**
```python
def check_endpoint_health(endpoint_name, test_message):
    """Real-time endpoint health validation using actual inference requests"""
    
def check_endpoint_status(endpoint_name):
    """Endpoint status monitoring for operational awareness"""
    
def verify_scaling_policy(endpoint_name):
    """Validates auto-scaling configuration and displays current settings"""
    
def check_scaling_activities(endpoint_name, MaxResults=50):
    """Historical scaling activity analysis for troubleshooting"""
```

**Advanced Rollback Capabilities:**
```python
def get_previous_config(endpoint_name):
    """Intelligent previous configuration detection using timestamp-based naming"""
    
def rollback_to_previous(endpoint_name, test_message):
    """
    One-click rollback to previous working configuration:
    1. Identifies previous endpoint configuration
    2. Performs rollback with scaling management
    3. Validates rollback success with health checks
    """
```

#### 3. Container Infrastructure

**Production-Grade Dockerfile:**
```dockerfile
# Optimized AWS Deep Learning Container
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04-sagemaker

WORKDIR /opt/ml/model

# Dependency management with version pinning
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy scipy pandas flask gunicorn pickle5
RUN pip install --no-cache-dir -v xgboost==2.1.1
RUN pip install --no-cache-dir joblib==1.4.2
RUN pip install --no-cache-dir scikit-learn

# Model artifact deployment
COPY proficiency_model.json /opt/ml/model/
COPY confidence_score_scaler.json /opt/ml/model/  
COPY confidence_model.json /opt/ml/model/
COPY item_params.csv /opt/ml/model/
COPY inference.py /opt/ml/model/

# Production WSGI server
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "inference:app"]
```

**High-Performance Inference Application (`inference.py`):**

*742-line Flask application* optimized for real-time educational ML inference:

**Model Loading and Initialization:**
```python
# Multi-model artifact loading
proficiency_model = xgb.Booster()
proficiency_model.load_model('proficiency_model.json')

confidence_model = xgb.Booster()  
confidence_model.load_model('confidence_model.json')

# Custom scaling and transformation utilities
class NanPreservingScaler:
    """Handles missing data gracefully during inference"""
    
class PercentileRankCalculator:
    """Converts confidence scores to percentile ranks for interpretability"""

# Item Response Theory parameters (38MB CSV with 50K+ questions)
item_params = pd.read_csv('item_params.csv')
```

**Real-Time Feature Engineering:**
```python
def process_inference_data(df, item_params):
    """
    Identical feature engineering pipeline to training:
    1. Correctness normalization (0-100 scale handling)
    2. Temporal feature calculation (time differences in hours)
    3. Duration capping and log transformation
    4. IRT parameter integration (11 lag positions)
    5. Spread feature calculation (max-min across history)
    6. Mean aggregations for discriminability and AUC-ROC
    7. Missing data handling with educational domain knowledge
    """
```

**Dual Processing Modes:**
```python
def process_input_original(data):
    """Original lag structure for skill-level predictions"""
    
def process_input_lagged(data):
    """Shifted lag structure for item-level predictions
    Handles API quirk where current question appears in history"""
```

**Production Endpoints:**
```python
@app.route('/ping', methods=['GET'])
def ping():
    """SageMaker health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Main inference endpoint supporting:
    - Single prediction requests
    - Batch prediction requests (multiple students)
    - Comprehensive input validation
    - Educational domain-specific error handling
    - Debug information for troubleshooting
    - Model version tracking
    """
```

### Production Deployment Workflow

#### Phase 1: Pre-Deployment Validation
1. **Artifact Verification**: Validate all model files, scalers, and item parameters
2. **Container Build**: Create optimized Docker image with dependency management
3. **Local Testing**: Comprehensive inference testing before cloud deployment
4. **ECR Management**: Automated repository creation and image versioning

#### Phase 2: SageMaker Deployment  
1. **Endpoint Assessment**: Check for existing endpoints and current configurations
2. **Model Registration**: Create SageMaker model with timestamped naming
3. **Configuration Management**: Generate endpoint configuration with scaling policies
4. **Rolling Deployment**: Zero-downtime updates or new endpoint creation

#### Phase 3: Production Optimization
1. **Auto-Scaling Setup**: Configure target tracking based on invocations per instance
2. **Health Monitoring**: Continuous endpoint health validation with real inference requests
3. **Performance Tuning**: Scaling parameter optimization for cost and latency
4. **Rollback Preparation**: Previous configuration preservation for rapid rollback

#### Phase 4: Operational Excellence
1. **Comprehensive Testing**: 15+ test scenarios covering edge cases and production patterns
2. **Scaling Verification**: Auto-scaling policy validation and activity monitoring
3. **Production Monitoring**: CloudWatch integration and performance metrics
4. **Incident Response**: Automated rollback capabilities and manual intervention tools

### Advanced Features

#### Intelligent Endpoint Management
- **Timestamped Versioning**: Automatic model and configuration versioning using datetime stamps
- **Previous Configuration Tracking**: Sophisticated previous version detection for rollbacks
- **Health-Check Driven Updates**: Automatic rollback if new deployments fail health validation
- **Scaling-Aware Updates**: Auto-scaling deregistration/reregistration during updates

#### Production Testing Framework
- **Malformed Input Resilience**: Tests for double JSON serialization and invalid formats
- **Batch Processing Validation**: Multi-student simultaneous prediction testing
- **Edge Case Coverage**: Empty histories, missing data, invalid skill IDs
- **Real Data Integration**: Testing with actual production student response patterns

#### Enterprise Monitoring
- **Real-Time Health Checks**: Continuous endpoint validation using domain-specific test cases
- **Scaling Activity Analysis**: Historical scaling event tracking and analysis
- **Performance Metrics**: Invocation rates, response times, and error rates
- **Configuration Auditing**: Current scaling policy and endpoint configuration reporting

### Usage Examples

**Complete Deployment Pipeline:**
```python
# Execute main deployment script
# Sets up variables, builds Docker, tests locally, deploys to SageMaker
%run BuildAndDeployDockerImageToSagemakerEndpoint.ipynb
```

**Manual Scaling Configuration:**
```python
from sagemaker_deploy_tools import setup_auto_scaling

setup_auto_scaling(
    endpoint_name='[REDACTED]',
    min_capacity=1,
    max_capacity=100, 
    target_value=15,  # invocations per instance per minute
    scale_in_cooldown=60,
    scale_out_cooldown=300
)
```

**Production Rollback:**
```python
from sagemaker_deploy_tools import rollback_to_previous

test_message = {"skillId": "test_skill", ...}
rollback_to_previous('[REDACTED]', test_message)
```

**Health and Scaling Monitoring:**
```python
from sagemaker_deploy_tools import verify_scaling_policy, check_scaling_activities

verify_scaling_policy('[REDACTED]')
check_scaling_activities('[REDACTED]', MaxResults=20)
```

### Integration with Educational ML Pipeline

The ModelImplementationWSDK seamlessly integrates with the broader educational assessment system:

- **Model Artifacts**: Deploys XGBoost models from ProficiencyModelTrainingPipeline
- **Feature Engineering**: Implements identical transformations to training pipeline
- **Item Parameters**: Integrates IRT/SDT parameters from ItemParametersCalculate
- **Real-Time Processing**: Handles live student response data from educational applications
- **Scalability**: Auto-scales based on student activity patterns and assessment demand

This comprehensive deployment framework ensures that sophisticated educational ML models can be reliably deployed and maintained in production environments with enterprise-grade operational capabilities.

---

## ItemParametersCalculate

### Overview
**ItemParametersCalculate** implements Item Response Theory (IRT) and Signal Detection Theory (SDT) for educational assessment. It analyzes student responses to calculate comprehensive item and student parameters that characterize question difficulty, discrimination, and student ability.

### Core Functionality

#### 1. IRT Parameter Estimation
Implements both 3-Parameter Logistic (3PL) and 4-Parameter Logistic (4PL) models:

**3PL Model Parameters:**
- **Discriminability (α)**: How well the item distinguishes between different ability levels (higher = better discrimination)
- **Difficulty (β)**: The ability level at which there's a 50% chance of correct response
- **Guessing (γ)**: Probability of getting the item correct by random guessing

**4PL Model Parameters (includes 3PL plus):**
- **Inattention (ε)**: Probability of getting the item wrong due to carelessness or inattention

#### 2. Signal Detection Theory Parameters
Additionally calculates SDT performance metrics:

- **AUC-ROC**: Area Under the ROC Curve - measures overall discriminatory power (0.5 = random, 1.0 = perfect)
- **Optimal Threshold**: Best cutoff point for classifying correct/incorrect responses
- **True Positive Rate (TPR)**: Sensitivity - correctly identifying capable students
- **True Negative Rate (TNR)**: Specificity - correctly identifying struggling students

#### 3. Key Functions

**`solve_IRT_for_matrix(table, iterations=50, FOUR_PL=True)`**
Main estimation function that:
- Takes student×item response matrix
- Iteratively estimates item parameters and student abilities
- Converges through maximum likelihood estimation
- Returns comprehensive IRTResults object

**`parallel_estimate_parameters_for_skill(table, thetas, FOUR_PL=True)`**
Parallelized parameter estimation for computational efficiency:
- Uses joblib for parallel processing
- Estimates parameters for all items simultaneously
- Handles missing data gracefully
- Returns parameter arrays and error estimates

**`export_object_to_csv(solvedIRT, skill_id, filename='estimatedItemParameters.csv')`**
Exports comprehensive parameter set including:
- All IRT parameters (discriminability, difficulty, guessing, inattention)
- Parameter estimation errors 
- SDT metrics (AUC-ROC, optimal threshold, TPR, TNR)
- Metadata (sample sizes, accuracy rates, creation dates)

### Usage Example

```python
import customPyIRT as irt

# Load student response data (students×items matrix)
response_data = load_response_matrix()

# Solve IRT model
results = irt.solve_IRT_for_matrix(
    response_data, 
    iterations=50, 
    FOUR_PL=True,
    verbose=True
)

# Export parameters for deployment
irt.export_object_to_csv(results, skill_id='ela_reading', filename='item_params.csv')

# Analyze parameter convergence
irt.plot_sample_parameter_convergence(results)
irt.timeCourseOfParameterConvergence(results)
```

### Mathematical Models

#### 3PL Logistic Function
```
P(correct) = γ + (1-γ) * (1 / (1 + exp(-α(θ-β))))
```

#### 4PL Logistic Function  
```
P(correct) = γ + (ε-γ) * (1 / (1 + exp(-α(θ-β))))
```

Where:
- θ (theta) = student ability
- α (alpha) = item discriminability  
- β (beta) = item difficulty
- γ (gamma) = guessing parameter
- ε (epsilon) = inattention parameter

### Output Data Structure

The exported CSV contains these fields for each question:
- `question_id`, `skill_id`
- `discriminability`, `difficulty`, `guessing`, `inattention`
- `discriminability_error`, `difficulty_error`, `guessing_error`, `inattention_error`
- `auc_roc`, `optimal_threshold`, `tpr`, `tnr`
- `skill_optimal_threshold`, `student_mean_accuracy`, `sample_size`
- `date_created`, `version`

---

## ProficiencyModelTrainingPipeline

### Overview
**ProficiencyModelTrainingPipeline** is a comprehensive machine learning pipeline that orchestrates the complete workflow from raw educational data to production-ready student proficiency prediction models. Built around XGBoost and custom educational ML components, it handles data processing, feature engineering, model training, evaluation, and inference testing with sophisticated educational domain-specific optimizations.

### Core Architecture

The pipeline is structured around a modular design with distinct phases controlled by boolean flags, allowing flexible execution of different pipeline stages.

#### 1. Pipeline Orchestration (`run_full_pipeline`)
The master function that coordinates all pipeline stages:

```python
def run_full_pipeline(
    pipeline_id='_v0',                    # Version identifier
    feature_set_id='_standardFeatures',   # Feature set identifier  
    model_type='_standard',               # Model variant identifier
    doSnowflakeETL=False,                 # Enable/disable ETL
    skillExplode=True,                    # Skill explosion mode
    CorrectnessBinary=False,              # Binary vs continuous target
    UseSigmoidTransform=False,            # Sigmoid-transformed XGBoost
    SigmoidTransformOutput=False,         # Apply sigmoid to outputs
    SigmoidTransformOutputFit=False,      # Fit sigmoid transformer
    ItemAgnostic=False,                   # Item-agnostic inference
    ItemAgnosticFit=False,                # Item-agnostic training
    ItemAgnosticDoubleFit=False,          # Double-fit item-agnostic
    F1ObjectiveMeasure=False,             # Use F1 objective
    pullNewData=False,                    # Fetch fresh data
    GenerateTestTrainSplit=False,         # Create data splits
    ItemParameterCalculate=False,         # Calculate IRT parameters
    FeatureEngineering=False,             # Feature engineering
    ProficiencyModelFit=False,            # Train proficiency model
    ConfidenceModelFit=False,             # Train confidence model
    InferenceModeOn=False,                # Run inference testing
    TestMode=False,                       # Limited data testing
    n_estimators=200,                     # XGBoost iterations
    FUTURE_WINDOW=0,                      # Future correctness window
    scalerFlag=False,                     # Enable feature scaling
    early_stopping_rounds=10,             # Early stopping
    INFERENCE_BATCH_SIZE=10000            # Inference batch size
):
```

#### 2. Data Management and Student ID Splitting

**Student ID Management:**
```python
def read_unique_studentids_from_parquet_files(file_pattern, column_names, batch_size=100):
    """
    Efficiently extracts unique student IDs from large parquet datasets
    - Uses multiprocessing for parallel file processing
    - Handles memory efficiently with batching
    - Returns pandas Series of unique student IDs
    """
```

**Data Splitting Strategy:**
- **Parameter IDs (50% of train)**: Used for IRT parameter calculation
- **Train IDs (45% total)**: Split into skill_train_ids and item_train_ids  
- **Test IDs (10% total)**: Held out for final evaluation
- Uses `train_test_split` with fixed random_state=42 for reproducibility

#### 3. Feature Engineering (`process_parquet_files`)

**Temporal Feature Engineering:**
```python
def process_parquet_files(input_pattern, output_folder, train_ids, item_params_file, 
                         column_names, TestMode, FUTURE_WINDOW):
    """
    Comprehensive feature engineering pipeline:
    
    1. Correctness normalization (divide by 100)
    2. Time difference calculations (hours between responses)
    3. Duration seconds capping (1-300 seconds)
    4. Log transformations for temporal features
    5. Item parameter integration (11 lags: current + 10 historical)
    6. Spread features (max-min across historical values)
    7. Mean aggregations for discriminability and AUC-ROC
    """
```

**Feature Categories Generated:**
- **Historical Correctness**: `CORRECTNESS_LAG_1` through `CORRECTNESS_LAG_10`
- **Temporal Features**: `OCCURREDAT_DIFF_1` through `OCCURREDAT_DIFF_10` (log-transformed)
- **Duration Features**: `LOG_DURATIONSECONDS_LAG_1` through `LOG_DURATIONSECONDS_LAG_10`
- **Item Parameters**: Difficulty, discriminability, guessing, inattention for each lag
- **Derived Features**: Spread features, mean aggregations, sample size logs
- **Future Labels**: `FUTURE_CORRECTNESS_LAG_1` through `FUTURE_CORRECTNESS_LAG_5` (when FUTURE_WINDOW > 0)

#### 4. Custom ML Components

**NanPreservingScaler:**
```python
class NanPreservingScaler:
    """
    Custom scaler that handles NaN values gracefully
    - Preserves NaN values during scaling
    - Uses incremental fitting for large datasets
    - Implements standard scaling (mean=0, std=1) on non-NaN values
    """
    def partial_fit(self, X):
        # Incremental mean and variance calculation
    
    def transform(self, X):
        # Scale while preserving NaN positions
```

**SigmoidXGBRegressor:**
```python
class SigmoidXGBRegressor:
    """
    XGBoost wrapper with built-in sigmoid transformation
    - Applies sigmoid to targets during training
    - Enables probability-based optimization
    - Maintains XGBoost interface compatibility
    """
    def train(self, dtrain, num_boost_round=100, ...):
        # Transform targets with sigmoid before training
    
    def predict(self, dmatrix):
        # Apply inverse sigmoid to predictions
```

**PercentileRankCalculator:**
```python
class PercentileRankCalculator:
    """
    Confidence score transformer using percentile ranking
    - Converts raw confidence scores to percentile ranks
    - Uses scipy.stats.rankdata for ranking
    - Saves/loads state for inference consistency
    """
```

#### 5. Model Training Functions

**Proficiency Model Training (`train_proficiency_model`):**
```python
def train_proficiency_model(input_folder, model_output_folder, CorrectnessBinary, 
                           TestMode=False, n_estimators=200, scalerFlag=False, 
                           early_stopping_rounds=10, UseSigmoidTransform=False,
                           SigmoidTransformOutput=False, F1ObjectiveMeasure=False,
                           ItemAgnosticFit=False, ItemAgnosticDoubleFit=False):
    """
    Main proficiency model training with XGBoost:
    
    1. Feature selection (excludes STUDENTID, SKILL, QUESTIONID_LAG_*, FUTURE_CORRECTNESS_*)
    2. Optional NanPreservingScaler fitting and application
    3. Incremental XGBoost training across parquet files
    4. Early stopping based on validation performance
    5. Model saving and feature importance logging
    """
```

**Key Training Features:**
- **Incremental Training**: Processes one parquet file per epoch to handle large datasets
- **Early Stopping**: Monitors test performance with configurable patience
- **Custom Metrics**: `reg_custom_metric` and `binary_custom_metric` for comprehensive evaluation
- **Multiple Objectives**: Support for F1, logistic, and squared error objectives

**Confidence Model Training (`train_confidence_model`):**
```python
def train_confidence_model(input_folder, model_output_folder, proficiency_model_file,
                          CorrectnessBinary, TestMode=False, FUTURE_WINDOW=0,
                          n_estimators=200, scalerFlag=False, UseSigmoidTransform=False,
                          SigmoidTransformOutput=False, ItemAgnosticFit=False):
    """
    Confidence model training process:
    
    1. Load proficiency model and generate predictions
    2. Add proficiency predictions as input features  
    3. Calculate absolute error as target variable
    4. Train XGBoost regressor to predict errors
    5. Fit PercentileRankCalculator for confidence score transformation
    """
```

**Skill Model Training (`train_skill_model`):**
```python
def train_skill_model(input_folder, model_output_folder, proficiency_model_file,
                     CorrectnessBinary, TestMode=False, FUTURE_WINDOW=0,
                     scalerFlag=False, UseSigmoidTransform=True, ItemAgnosticFit=False):
    """
    Fits sigmoid transformation for skill-level predictions:
    
    1. Generate proficiency model predictions
    2. Calculate targets using future correctness window
    3. Optimize sigmoid parameters (a, b) using scipy.optimize
    4. Save sigmoid parameters for inference
    """
```

#### 6. Evaluation and Testing Framework

**Custom Metrics Implementation:**
```python
def reg_custom_metric(preds, dtrain, triggerFull=False, DirectCall=False, 
                     tgt_threshold=0.5, pred_threshold=0.5, label=None):
    """
    Comprehensive regression metrics including:
    - RMSE, MAE, R²
    - AUC-ROC, AUC-PR  
    - Accuracy, Precision, Recall, F1
    - Calibration metrics
    """

def binary_custom_metric(preds, dtrain, triggerFull=False, DirectCall=False,
                        tgt_threshold=None, pred_threshold=0.5, label=None):
    """
    Binary classification metrics optimized for educational assessment
    """
```

**Inference Testing (`InferenceTesting`):**
```python
def InferenceTesting(input_folder, proficiency_model_output_folder, 
                    confidence_model_output_folder, item_params_file,
                    CorrectnessBinary, TestMode=False, FUTURE_WINDOW=0,
                    ItemAgnostic=False, ItemAgnosticDoubleFit=False,
                    BATCH_SIZE=10000, smallInferenceRun=None):
    """
    Comprehensive inference testing with multiple model variants:
    
    1. Item-Aware Models (use next question information)
    2. Item-Agnostic Models (no next question information) 
    3. Skill Models (sigmoid-transformed predictions)
    
    Each variant includes:
    - Proficiency prediction
    - Confidence prediction  
    - Confidence score (percentile rank)
    """
```

#### 7. Integration with ItemParametersCalculate

**IRT Parameter Integration (`process_skills_irtsdt`):**
```python
def process_skills_irtsdt(file_pattern, column_names, output_filename, version, train_ids,
                         binary=False, restart_index=0, TestMode=False, DATA_LIMIT=None):
    """
    Orchestrates IRT parameter calculation:
    
    1. Extract unique skills from parquet files
    2. Process each skill separately using multiprocessing
    3. Call customPyIRT for parameter estimation
    4. Export results with resumable calculation
    """
```

#### 8. Data Format Conversion and API Testing

**JSON Conversion for Live Testing:**
```python
def transform_row_to_json(row, includeCurrent=True):
    """
    Converts feature-engineered data back to API input format:
    
    1. Extracts question IDs, correctness, durations from lag columns
    2. Formats timestamps to match API requirements
    3. Handles null values by shortening history arrays
    4. Creates JSON structure matching live inference API
    """
```

### Configuration and Output Management

#### Pipeline Configuration
The pipeline uses extensive boolean flags for stage control:

- **Data Pipeline**: `doSnowflakeETL`, `pullNewData`, `GenerateTestTrainSplit`
- **Model Training**: `ProficiencyModelFit`, `ConfidenceModelFit`, `SigmoidTransformOutputFit`
- **Model Variants**: `UseSigmoidTransform`, `SigmoidTransformOutput`, `ItemAgnosticFit`
- **Evaluation**: `InferenceModeOn`, `InferenceModeFeatureEngineering`

#### Output Artifacts

**Model Files:**
- `xgb_model.json` - Proficiency XGBoost model
- `confidence_xgb_model.json` - Confidence XGBoost model
- `sigmoid_params.json` - Sigmoid transformation parameters
- `nan_preserving_scaler_proficiency.joblib` - Feature scaler
- `percentile_rank_calculator.pkl` - Confidence score transformer

**Feature and Metadata:**
- `feature_names.json` - Model input features
- `confidence_feature_names.json` - Confidence model features  
- `feature_engineered_column_names.json` - All engineered features
- Training logs with comprehensive metrics and sample predictions

**Performance Visualizations:**
- Model output distribution plots
- Error distribution analysis
- Error by response count analysis
- Feature importance rankings

### Advanced Features

#### Multi-Model Architecture
The pipeline trains multiple model variants simultaneously:

1. **Item-Aware Models**: Use next question characteristics for prediction
2. **Item-Agnostic Models**: Ignore next question information (more generalizable)
3. **Skill Models**: Aggregate item predictions to skill-level estimates

#### Future Window Modeling
Uses `FUTURE_WINDOW` parameter to train confidence models on multi-step ahead prediction errors, providing more robust confidence estimation.

#### Memory-Efficient Processing
- Incremental model training to handle large datasets
- Parquet-based data storage for efficient I/O
- Multiprocessing for parallel data processing
- Configurable batch sizes for inference testing

#### 9. Inference Code Export and Testing (`InferenceCodeTestAndWrite.ipynb`)

This crucial component bridges the gap between trained models and production deployment by generating and testing production-ready inference code:

**Primary Functions:**

**Inference Code Generation (`writeOutInferenceFile`):**
```python
def export_cell_with_flask_wrapper(cell_index, output_path):
    """
    Exports a notebook cell with Flask wrapper for SageMaker deployment
    
    Generates complete inference.py file with:
    - Flask application framework
    - Model loading and initialization
    - Input validation and preprocessing
    - Feature engineering pipeline
    - Multi-model prediction coordination
    - Error handling and debug information
    - SageMaker-compatible endpoints (/ping, /invocations)
    """
```

**Output Location and Integration:**
The generated inference code is exported to:
```
../ModelImplementationWSDK/ELA_student_proficiency_model_v0/container/inference.py
```

This directly integrates with the ModelImplementationWSDK for SageMaker deployment.

**Generated Inference Script Features:**
- **Model Initialization**: Loads XGBoost models, item parameters, scalers, and confidence transformers
- **Input Processing**: Validates and transforms API requests into feature vectors
- **Feature Engineering**: Real-time implementation of training pipeline transformations
- **Multi-Model Inference**: Coordinates proficiency, confidence, and skill predictions
- **Error Handling**: Comprehensive exception handling with detailed debug information
- **SageMaker Endpoints**: 
  - `/ping` - Health check endpoint
  - `/invocations` - Main prediction endpoint supporting both single and batch requests

**Comprehensive Testing Framework:**

**1. Synthetic Data Generation:**
```python
def generate_sample_input(num_history=10):
    """
    Creates realistic test inputs with:
    - Valid and invalid skill IDs for edge case testing
    - Random question sequences and performance patterns
    - Proper timestamp formatting
    - Variable history lengths
    """
```

**2. Input Structure Validation:**
```python
def verify_input_structure(input_data):
    """
    Validates API input format:
    - Required fields: skillId, questionId, eventTime
    - History arrays: questionIdsHistory, correctnessHistory, durationSecondsHistory, eventTimesHistory
    - Proper data types and formats
    - Single vs batch input handling
    """
```

**3. Inference Function Testing:**
```python
def run_inference(input_data):
    """
    Core inference testing with:
    - Single input processing
    - Batch input processing  
    - Valid skill ID handling
    - Invalid skill ID graceful degradation
    - Feature engineering verification
    - Output format validation
    """
```

**4. Output Format Verification:**
The testing validates that inference produces the expected output structure:
```python
# Expected output format
{
    'item_prediction': [0.488109],           # Proficiency probability
    'item_prediction_error': [0.267804],     # Confidence model prediction
    'item_prediction_confidence': [27.301],  # Percentile rank confidence
    'skill_prediction': [0.472],             # Skill-level prediction (when available)
    'skill_prediction_error': [0.245],       # Skill-level confidence  
    'skill_prediction_confidence': [31.2]    # Skill-level percentile confidence
}
```

**5. Performance and Edge Case Testing:**
- **Valid Skills**: Tests with known skill IDs for full prediction pipeline
- **Invalid Skills**: Tests graceful degradation with fake skill IDs (returns item-level predictions only)
- **Variable History**: Tests with different history lengths (1-10 previous responses)
- **Missing Data**: Tests handling of incomplete history data
- **Batch Processing**: Tests multiple inputs simultaneously
- **Response Time**: Validates sub-second inference latency

**Integration with Deployment Pipeline:**

**Artifact Management:**
```python
# Moves all model artifacts to deployment directory
artifact_destination_dir = '../ModelImplementationWSDK/ELA_student_proficiency_model_v0/container/'

# Files copied for deployment:
# - xgb_model.json (proficiency model)
# - confidence_xgb_model.json (confidence model)  
# - percentile_rank_calculator.pkl (confidence transformer)
# - ELA_ItemParameters_v5.csv (IRT parameters)
# - feature_names.json (model input specification)
```

**Code Export Process:**
```python
if writeOutInferenceFile:
    # Export inference cell with Flask wrapper
    current_cell = len(In) - 2
    export_cell_with_flask_wrapper(current_cell, artifact_destination_dir + 'inference.py')
    print('Exported inference.py with Flask wrapper and inference code.')
```

**Debug and Validation Features:**
- **Input Inspection**: Option to return processed input alongside predictions for debugging
- **Feature Vector Logging**: Detailed logging of feature engineering transformations
- **Model Version Tracking**: Embedded version information for production monitoring
- **Comprehensive Error Messages**: Detailed error reporting for troubleshooting

This testing framework ensures that the inference code:
1. Handles all expected input formats correctly
2. Produces consistent outputs across different scenarios
3. Gracefully handles edge cases and invalid inputs
4. Integrates seamlessly with SageMaker deployment infrastructure
5. Maintains identical behavior between local testing and production deployment

### Usage Examples

**Full Pipeline Execution:**
```python
# Complete training pipeline
run_full_pipeline(
    pipeline_id='_v60',
    feature_set_id='_v50IP',
    doSnowflakeETL=True,
    pullNewData=True,
    GenerateTestTrainSplit=True,
    ItemParameterCalculate=True,
    FeatureEngineering=True,
    ProficiencyModelFit=True,
    ConfidenceModelFit=True,
    SigmoidTransformOutputFit=True,
    InferenceModeOn=True,
    TestMode=False,
    n_estimators=200,
    FUTURE_WINDOW=5
)
```

**Inference-Only Mode:**
```python
# Test existing models
run_full_pipeline(
    pipeline_id='_v60',
    InferenceModeOn=True,
    InferenceModeFeatureEngineering=True,
    TestMode=False,
    INFERENCE_BATCH_SIZE=10000
)
```

---

## WithinSkill_ELA

### Overview
**WithinSkill_ELA** serves as a comprehensive reference implementation demonstrating the complete integration of all system components for English Language Arts (ELA) proficiency prediction. This project contains working models, testing frameworks, and deployment artifacts that showcase the practical application of SnowflakeETL, ItemParametersCalculate, ProficiencyModelTrainingPipeline, and ModelImplementationWSDK.

### Project Structure

```
WithinSkill_ELA/
├── model_v60/                                          # Latest model version
│   ├── ELA_Proficiency_v6.ipynb                      # Model training notebook
│   ├── proficiency_model_vAllData01042025_v50IP_doublefit_sigmoid/  # Proficiency model
│   ├── confidence_model_vAllData01042025_v50IP_doublefit_sigmoid/   # Confidence model
│   └── feature_engineered_train_parquets_v50IP/      # Training data
├── model_v40_fullBuild/                               # Previous model version
├── InferenceCodeTestAndWrite.ipynb                   # Inference wrapper creation
├── LiveModelTest.ipynb                               # Production testing
└── model_validation_outputs/                         # Test results and metrics
```

### Core Components

#### 1. Model Training (`model_v60/ELA_Proficiency_v6.ipynb`)
The primary training notebook that demonstrates a complete model development lifecycle using ProficiencyModelTrainingPipeline:

**Current Model Configuration (v60):**
```python
doSnowflakeETL = False           # Using pre-processed data
ItemAgnosticDoubleFit = True     # Enhanced training approach  
SigmoidTransformOutput = False   # Direct XGBoost outputs
FUTURE_WINDOW = 0               # Next-item prediction
InferenceModeOn = True          # Testing existing models
TestMode = False                # Full evaluation mode
```

**Model Architecture:**
- **Proficiency Model**: XGBoost model at `proficiency_model_vAllData01042025_v50IP_doublefit_sigmoid/xgb_model.json`
- **Confidence Model**: Separate XGBoost model at `confidence_model_vAllData01042025_v50IP_doublefit_sigmoid/confidence_xgb_model.json`
- **Score Transformation**: Percentile rank calculator for confidence scaling at `confidence_model_vAllData01042025_v50IP_doublefit_sigmoid/percentile_rank_calculator.pkl`

**Data Sources:**
- **Training Data**: Pre-processed parquet files from `feature_engineered_train_parquets_v50IP/`
- **Item Parameters**: `ELA_ItemParameters_v5.csv` containing IRT analysis results
- **Test Data**: Reserved test sets from `inferenceTest_feature_engineered_train_parquets_v50IP/`

**Real Performance Metrics** (from actual test runs):
- **Item-Aware Proficiency**: MSE ~0.173, RMSE ~0.416, R² ~0.217, AUC-ROC ~0.774-0.800
- **Item-Agnostic Proficiency**: MSE ~0.192, RMSE ~0.438, R² ~0.126, AUC-ROC ~0.700-0.738  
- **Confidence Estimation**: MSE ~0.325, AUC-ROC ~0.555 (challenging prediction task)
- **Threshold Parameters**: Struggling=0.481, Proficiency=0.700

#### 2. Live Model Validation (`LiveModelTest.ipynb`)
Production model validation system that compares local inference with deployed SageMaker endpoints:

**Validation Process:**
1. **Data Extraction**: Queries recent production predictions from Snowflake:
   ```sql
   SELECT * FROM PROGRESS.ELA_PROFICIENCY_EVENTS 
   WHERE OCCURREDAT >= '2024-12-28'
   AND METADATA:sagemakerResponse:modelVersion[0] = '3.9c14c'
   ```

2. **Local Environment Recreation**: 
   - Copies production container files from `../ModelImplementationWSDK/ELA_student_proficiency_model_v0/container/`
   - Loads identical models and artifacts
   - Recreates inference pipeline locally using identical code

3. **Prediction Comparison**: 
   - Processes identical inputs through local and production models
   - Validates prediction consistency across production data
   - Generates detailed comparison metrics and identifies discrepancies

**Key Testing Results:**
- Production consistency validation across thousands of real student interactions
- Model version tracking (currently testing version 3.9c14c)
- Response format validation for SageMaker integration
- Performance benchmarking for production latency requirements

#### 3. Inference Development (`InferenceCodeTestAndWrite.ipynb`)
Comprehensive inference pipeline development and testing environment that generates production-ready deployment code:

**Core Functions:**
- **Model Loading**: Initializes XGBoost models, item parameters, and transformation artifacts
- **Input Validation**: Ensures proper format for real-time student data
- **Feature Engineering**: Real-time transformation using identical pipeline to training
- **Multi-Model Coordination**: Integrates proficiency and confidence predictions
- **Inference Code Export**: Generates production-ready Flask application

**Input Data Format:**
```json
{
  "skillId": "1610f61e-e69f-e311-9503-005056801da1",
  "questionId": "question_618", 
  "eventTime": "2025-01-03T04:28:10.058137",
  "questionIdsHistory": ["question_627", "question_513", "question_221"],
  "correctnessHistory": [0, 100, 0],
  "durationSecondsHistory": [205, 17, 34],
  "eventTimesHistory": ["2025-01-03T04:28:10.058137", ...]
}
```

**Generated Inference Code (`writeOutInferenceFile` function):**
Exports a complete Flask application to `../ModelImplementationWSDK/ELA_student_proficiency_model_v0/container/inference.py` including:
- Model initialization and loading logic
- Input parsing and validation  
- Feature engineering pipeline (identical to training)
- Multi-model prediction coordination
- Comprehensive error handling and debugging
- Health check endpoints (`/ping`)
- Prediction endpoints (`/invocations`)

#### 4. Model Artifacts

**Production Model Files:**
```
model_v60/
├── proficiency_model_vAllData01042025_v50IP_doublefit_sigmoid/
│   └── xgb_model.json                    # Main proficiency XGBoost model
├── confidence_model_vAllData01042025_v50IP_doublefit_sigmoid/
│   ├── confidence_xgb_model.json         # Confidence estimation XGBoost model
│   └── percentile_rank_calculator.pkl   # Score transformation utility
└── ELA_ItemParameters_v5.csv           # IRT parameters from ItemParametersCalculate
```

**Training Data:**
- **823 parquet files** in `feature_engineered_train_parquets_v50IP/` containing processed student response data
- **529,397 studentids** for training split
- **264,699 studentids** for test split
- **Column structure**: 52 columns including STUDENTID, SKILL, QUESTIONID, OCCURREDAT, CORRECTNESS, DURATIONSECONDS, lag features, and future labels

### Real-World Performance

#### Production Integration Results:
- **Model Version**: 3.9c14c currently deployed in production
- **Test Coverage**: Validation across 30 test files with 25K-50K rows each
- **Accuracy**: Item-aware predictions achieve AUC-ROC of 0.774-0.800 consistently
- **Response Format**: Compatible with SageMaker endpoint requirements
- **Inference Speed**: Real-time processing suitable for production deployment

#### Model Evaluation Approach:
- **Reserved Test Sets**: Small inference run mode testing on 30 files with sufficient data
- **Item-Agnostic Testing**: Future prediction scenarios with next-item information erased
- **Multiple Prediction Types**: item_AWARE_prof, item_AGNOSTIC_prof, skill_prof, confidence, and correctness_surprise metrics
- **Threshold Optimization**: Precision/recall optimization for educational decision making

### Usage Patterns

#### For Model Development:
1. **Training**: Execute `ELA_Proficiency_v6.ipynb` with ProficiencyModelTrainingPipeline
2. **Testing**: Validate using inference testing framework in notebook
3. **Deployment**: Export inference code via `InferenceCodeTestAndWrite.ipynb`

#### For Production Monitoring:
1. **Live Validation**: Run `LiveModelTest.ipynb` to compare production vs. local predictions
2. **Performance Tracking**: Monitor prediction accuracy and model drift using Snowflake queries
3. **Version Control**: Track model versions through metadata in production predictions

#### For Research and Development:
1. **Inference Experimentation**: Use `InferenceCodeTestAndWrite.ipynb` for rapid prototyping
2. **Feature Analysis**: Examine model behavior through comprehensive test outputs
3. **Integration Testing**: Validate end-to-end pipeline from data extraction to deployment

### Integration Architecture

This implementation demonstrates the practical integration of all system components:
- **SnowflakeETL**: Historical data extraction for training and real-time validation data queries
- **ItemParametersCalculate**: ELA_ItemParameters_v5.csv provides IRT parameters for feature engineering  
- **ProficiencyModelTrainingPipeline**: Complete training pipeline with XGBoost models and evaluation
- **ModelImplementationWSDK**: Production deployment container with generated inference.py

The WithinSkill_ELA project serves as both a working production system and a comprehensive reference implementation for building scalable educational assessment ML pipelines.

---

## SnowflakeETL

### Overview
**SnowflakeETL** is a critical data extraction, transformation, and loading (ETL) component that interfaces with Snowflake data warehouse to pull and process raw educational response data. It serves as the foundation for the entire ML pipeline by providing clean, structured datasets that feed into item parameter calculation and model training processes.

### Core Functionality

#### 1. Data Extraction from Snowflake
SnowflakeETL connects to the [REDACTED] production Snowflake instance to extract student response data from multiple tables:

**Primary Data Sources:**
- `PROGRESS.ELA_ANSWER_EVENTS` - Core student response events with question answers, timing, and metadata
- `PROGRESS.ELA_PROFICIENCY_EVENTS` - Model prediction events for live testing and validation
- Associated skill and question mapping tables for enrichment

**Connection Management:**
- Handles Snowflake authentication and connection pooling
- Manages query timeouts and connection retries
- Supports both batch and streaming data extraction modes

#### 2. Three-Stage ETL Process
The ETL process is structured as a sequence of three SQL queries that progressively transform and enrich the data:

**Stage 1: Question-Skill Association Creation**
```sql
-- Creates association table linking questions to their skills
CREATE OR REPLACE TABLE [REDACTED].PRODUCT_TESTING.ELA_QUESTIONID_SKILLID_ASSOCIATIONS AS
SELECT DISTINCT
    QUESTIONID,
    ARRAY_COMPACT(
        ARRAY_CAT(
            COALESCE(FIRST_VALUE(QUESTION_SKILLS) OVER (...), ARRAY_CONSTRUCT()),
            ARRAY_CONSTRUCT(FIRST_VALUE(RLSKILLID) OVER (...))
        )
    ) AS UNIQUE_SKILLS
FROM PROGRESS.ELA_ANSWER_EVENTS
WHERE OCCURREDAT >= '2024-05-01'
```

**Stage 2: Skill Explosion (Configurable)**
Two modes available based on `skillExplode` flag:

*Exploded Mode (skillExplode=True):*
- Flattens question-skill associations to create one row per question-skill combination
- Enables skill-specific analysis and modeling
- Used for training separate models per skill

*Non-Exploded Mode (skillExplode=False):*
- Maintains original question structure without skill explosion
- Used for question-level modeling across multiple skills
- Preserves natural question-skill relationships

**Stage 3: Temporal Feature Engineering**
```sql
-- Adds historical context and future labels for each response
CREATE OR REPLACE TABLE ... AS
WITH ranked_events AS (
    SELECT *, 
           ROW_NUMBER() OVER (PARTITION BY STUDENTID, SKILL ORDER BY OCCURREDAT ASC) AS event_rank
    FROM previous_stage
)
SELECT 
    -- Current response
    STUDENTID, SKILL, QUESTIONID, OCCURREDAT, CORRECTNESS, DURATIONSECONDS,
    -- Historical features (10 previous responses)
    l1.QUESTIONID AS QUESTIONID_LAG_1,
    l1.CORRECTNESS AS CORRECTNESS_LAG_1,
    l1.DURATIONSECONDS AS DURATIONSECONDS_LAG_1,
    -- ... up to LAG_10
    -- Future labels (5 forward-looking responses for confidence modeling)
    f1.CORRECTNESS AS FUTURE_CORRECTNESS_LAG_1,
    -- ... up to LAG_5
FROM ranked_events
```

#### 3. Data Processing Functions

**Sequential Query Execution (`run_snowflake_queries_sequentially`)**
```python
def run_snowflake_queries_sequentially(queries):
    """
    Executes a sequence of SQL queries in order
    - Handles transaction management
    - Provides query execution logging
    - Manages error handling and rollback
    """
```

**Bulk Data Extraction (`grabAllDataFromSnowflake`)**
```python
def grabAllDataFromSnowflake(stage, query_file, datastore, csvTrigger=False):
    """
    Extracts large datasets from Snowflake to local parquet files
    
    Parameters:
    - stage: Snowflake stage for data export
    - query_file: SQL query defining data selection
    - datastore: Local directory for storing extracted data
    - csvTrigger: Whether to export as CSV (default: parquet)
    """
```

**Live Query Execution (`run_snowflake_query`)**
```python
def run_snowflake_query(query):
    """
    Executes single queries for real-time data retrieval
    - Used for live model testing and validation
    - Returns both column names and result data
    - Handles query result formatting
    """
```

#### 4. Data Transformation Capabilities

**Temporal Windowing:**
- Creates sliding windows of student response history (typically 10 previous responses)
- Generates forward-looking labels for confidence model training
- Handles variable-length sequences with proper null handling

**Skill-Level Aggregation:**
- Groups responses by student-skill combinations
- Maintains chronological ordering within skill progressions
- Supports both individual question analysis and skill-level modeling

**Metadata Enrichment:**
- Adds response timing information (duration, session position)
- Includes question difficulty and discrimination parameters
- Preserves original response context and student progression

#### 5. Integration with Training Pipeline

**Data Flow Integration:**
```python
# In ProficiencyModelTrainingPipeline
if doSnowflakeETL:
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
    
    # Execute ETL pipeline
    SnowflakeETL.run_snowflake_queries_sequentially(queries)
    
    # Extract processed data
    SnowflakeETL.grabAllDataFromSnowflake(
        stage='[REDACTED]',
        query_file=query_file,
        datastore=raw_data_folder_path,
        csvTrigger=False
    )
```

**Usage by Other Components:**

*ItemParametersCalculate Integration:*
- Provides cleaned response data for IRT/SDT analysis
- Ensures proper student-question-skill associations
- Maintains response chronology for parameter estimation

*ProficiencyModelTrainingPipeline Integration:*
- Supplies feature-engineered datasets for model training
- Provides train/test splits with temporal consistency
- Enables both batch and incremental model updates

*LiveModelTest Integration:*
- Enables real-time querying for live model validation
- Provides production data for A/B testing
- Supports model performance monitoring

### Configuration Options

#### ETL Pipeline Control
- `doSnowflakeETL`: Boolean flag to enable/disable ETL execution
- `skillExplode`: Controls whether to explode skills (skill-level vs question-level modeling)
- `pullNewData`: Whether to fetch fresh data or use existing cached data

#### Data Selection Parameters
- Date ranges for response filtering (typically last 6-12 months)
- Student population filters (active users, grade levels, etc.)
- Question type filters (multiple choice, open response, etc.)

#### Output Format Options
- Parquet format (default) for efficient ML pipeline processing
- CSV format for data analysis and debugging
- Compressed storage for large datasets

### Performance Optimizations

**Query Optimization:**
- Uses partitioned tables and clustered indexes
- Implements efficient window functions for temporal features
- Leverages Snowflake's columnar storage for aggregations

**Data Transfer Efficiency:**
- Utilizes Snowflake stages for bulk data movement
- Implements parallel data extraction for large datasets
- Supports incremental updates to reduce data transfer

**Storage Management:**
- Implements parquet format for efficient storage and loading
- Uses compression to minimize disk usage
- Maintains data versioning for reproducibility

### Monitoring and Logging

**ETL Process Monitoring:**
- Logs query execution times and row counts
- Tracks data freshness and update frequencies
- Monitors data quality metrics and anomalies

**Error Handling:**
- Comprehensive exception handling for connection issues
- Query retry logic for transient failures
- Data validation checks after extraction

### Usage Examples

**Full ETL Pipeline Execution:**
```python
# Configure ETL parameters
doSnowflakeETL = True
skillExplode = True
pullNewData = True

# Execute full pipeline
run_full_pipeline(
    doSnowflakeETL=doSnowflakeETL,
    skillExplode=skillExplode,
    pullNewData=pullNewData,
    raw_data_folder_path='StudentProficiencyData_ELA_09252024'
)
```

**Live Data Query:**
```python
# Query recent production data
query = """
SELECT * FROM PROGRESS.ELA_PROFICIENCY_EVENTS 
WHERE OCCURREDAT >= '2024-12-28'
AND METADATA:sagemakerResponse:modelVersion[0]= '3.9c14c'
ORDER BY OCCURREDAT DESC
"""

column_names, live_data = SnowflakeETL.run_snowflake_query(query)
```

**Incremental Data Update:**
```python
# Update with recent data only
SnowflakeETL.grabAllDataFromSnowflake(
    stage='[REDACTED]',
    query_file='incremental_query.txt',
    datastore='IncrementalUpdate_20250103',
    csvTrigger=False
)
```

---

## Integration Workflow

The five components work together in this typical workflow:

1. **SnowflakeETL**: Extract and transform raw student response data from Snowflake warehouse
2. **ItemParametersCalculate**: Analyze responses to extract item and student parameters using IRT/SDT
3. **ProficiencyModelTrainingPipeline**: Train ML models using ETL data + item parameters
4. **ModelImplementationWSDK**: Deploy trained models to production SageMaker endpoints
5. **WithinSkill_ELA**: Reference implementation showing complete pipeline integration

Each component is designed to be modular and reusable across different educational domains while providing comprehensive functionality for educational assessment and prediction. 
