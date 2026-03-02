# Educational Assessment ML System - Architecture Diagram

```mermaid
graph TB
    %% Data Sources
    SW[Snowflake Data Warehouse<br/>- PROGRESS.ELA_ANSWER_EVENTS<br/>- PROGRESS.ELA_PROFICIENCY_EVENTS<br/>- Student Response History]
    
    %% SnowflakeETL Component
    subgraph SETL["SnowflakeETL"]
        SE1[Query 1: Question-Skill Associations]
        SE2[Query 2: Skill Explosion/Non-Explosion]
        SE3[Query 3: Temporal Feature Engineering]
        SE4[Data Extraction to Parquet Files]
    end
    
    %% ItemParametersCalculate Component  
    subgraph IPC["ItemParametersCalculate"]
        IP1[IRT Parameter Estimation<br/>- 3PL/4PL Models<br/>- Discriminability, Difficulty<br/>- Guessing, Inattention]
        IP2[SDT Parameter Calculation<br/>- AUC-ROC, Optimal Threshold<br/>- TPR, TNR]
        IP3[Export Item Parameters CSV]
    end
    
    %% ProficiencyModelTrainingPipeline Component
    subgraph PMTP["ProficiencyModelTrainingPipeline"]
        PM1[Data Management<br/>- Student ID Splitting<br/>- Train/Test Split Generation]
        PM2[Feature Engineering<br/>- Temporal Features<br/>- IRT Parameter Integration<br/>- Lag Features]
        PM3[Model Training<br/>- XGBoost Proficiency Model<br/>- XGBoost Confidence Model<br/>- Sigmoid Transformation]
        PM4[Model Evaluation<br/>- Custom Metrics<br/>- Performance Validation]
        PM5[Inference Testing<br/>- Item-Aware/Agnostic Models<br/>- Batch Processing]
    end
    
    %% ModelImplementationWSDK Component
    subgraph WSDK["ModelImplementationWSDK"]
        WS1[Docker Container Build<br/>- Flask Inference App<br/>- Model Artifact Loading<br/>- ECR Push]
        WS2[Local Testing<br/>- Container Validation<br/>- Inference Testing]
        WS3[SageMaker Deployment<br/>- Endpoint Creation/Update<br/>- Auto-scaling Setup<br/>- Health Monitoring]
        WS4[Production Management<br/>- Rolling Updates<br/>- Automatic Rollback<br/>- Scaling Monitoring]
    end
    
    %% WithinSkill_ELA Reference Implementation
    subgraph WSE["WithinSkill_ELA"]
        WE1[Model Training<br/>ELA_Proficiency_v6.ipynb]
        WE2[Inference Development<br/>InferenceCodeTestAndWrite.ipynb]
        WE3[Live Validation<br/>LiveModelTest.ipynb]
        WE4[Production Artifacts<br/>- Proficiency Models<br/>- Confidence Models<br/>- Feature Transformers]
    end
    
    %% Production Environment
    subgraph PROD["Production Environment"]
        SM[AWS SageMaker Endpoint<br/>- Real-time Inference<br/>- Auto-scaling<br/>- Health Monitoring]
        CW[CloudWatch<br/>- Metrics & Alarms<br/>- Performance Monitoring]
        API[Student Assessment API<br/>- Live Predictions<br/>- Confidence Scores]
    end
    
    %% Data Flow
    SW --> SE1
    SE1 --> SE2
    SE2 --> SE3
    SE3 --> SE4
    
    SE4 --> IP1
    IP1 --> IP2
    IP2 --> IP3
    
    SE4 --> PM1
    IP3 --> PM2
    PM1 --> PM2
    PM2 --> PM3
    PM3 --> PM4
    PM4 --> PM5
    
    PM3 --> WS1
    PM5 --> WS1
    WS1 --> WS2
    WS2 --> WS3
    WS3 --> WS4
    
    %% Reference Implementation Integration
    SE4 -.-> WE1
    IP3 -.-> WE1
    WE1 --> WE2
    WE2 --> WE4
    WE4 --> WS1
    WE3 -.-> SM
    
    %% Production Deployment
    WS3 --> SM
    WS4 --> SM
    SM --> CW
    SM --> API
    
    %% Feedback Loop
    API -.-> SW
    CW -.-> WS4
    
    %% Styling
    classDef dataSource fill:#e1f5fe
    classDef etlComponent fill:#f3e5f5
    classDef analysisComponent fill:#e8f5e8
    classDef trainingComponent fill:#fff3e0
    classDef deploymentComponent fill:#fce4ec
    classDef referenceComponent fill:#f1f8e9
    classDef productionComponent fill:#ffebee
    
    class SW dataSource
    class SETL etlComponent
    class IPC analysisComponent
    class PMTP trainingComponent
    class WSDK deploymentComponent
    class WSE referenceComponent
    class PROD productionComponent
```

## Component Descriptions

### 1. **SnowflakeETL** (Data Foundation)
- **Purpose**: Extract and transform raw educational response data
- **Key Features**: Three-stage ETL process, skill explosion, temporal windowing
- **Output**: Clean parquet files for training and analysis

### 2. **ItemParametersCalculate** (Educational Analytics)  
- **Purpose**: Calculate item difficulty and discrimination using IRT/SDT
- **Key Features**: 3PL/4PL models, signal detection theory, parallel processing
- **Output**: Item parameter CSV with educational psychometric properties

### 3. **ProficiencyModelTrainingPipeline** (ML Engine)
- **Purpose**: Train sophisticated ML models for student proficiency prediction
- **Key Features**: XGBoost models, confidence estimation, feature engineering
- **Output**: Trained models and evaluation metrics

### 4. **ModelImplementationWSDK** (Production Deployment)
- **Purpose**: Deploy and manage ML models in production AWS infrastructure
- **Key Features**: Docker containers, auto-scaling, rolling updates, health monitoring
- **Output**: Live SageMaker endpoints with enterprise-grade operations

### 5. **WithinSkill_ELA** (Reference Implementation)
- **Purpose**: Demonstrate complete system integration for English Language Arts
- **Key Features**: End-to-end pipeline, inference testing, production validation
- **Output**: Working production models and deployment artifacts

## Data Flow Summary

1. **Data Ingestion**: Student responses flow from educational applications to Snowflake
2. **ETL Processing**: SnowflakeETL transforms raw data into ML-ready datasets
3. **Item Analysis**: ItemParametersCalculate generates educational psychometric parameters
4. **Model Training**: ProficiencyModelTrainingPipeline creates prediction models
5. **Deployment**: ModelImplementationWSDK deploys models to production infrastructure
6. **Live Inference**: Production endpoints serve real-time student proficiency predictions
7. **Monitoring**: Continuous performance monitoring and automatic scaling based on demand
8. **Feedback**: Production predictions and performance data inform future model iterations 