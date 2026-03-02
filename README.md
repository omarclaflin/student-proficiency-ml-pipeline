# Project Structure and Workflow

This repository contains a production ML system for estimating student proficiency in real time. It covers the full lifecycle: pulling student response data from a data warehouse via an ETL layer, computing psychometric item parameters using Item Response Theory (IRT) and Signal Detection Theory (SDT), training XGBoost-based proficiency and confidence models with temporal feature engineering, and deploying those models as auto-scaling inference endpoints on AWS SageMaker inside Docker containers. The pipeline includes automated model evaluation, inference wrapper generation, rolling endpoint updates with health checks and automatic rollback, and tools for monitoring scaling behavior in production. It was designed to serve real-time proficiency predictions for an adaptive learning platform across both Math and ELA subjects.

This was a solo project built and maintained by a single ML engineer / applied scientist throughout 2024, covering research, data engineering, model development, deployment infrastructure, and production monitoring. It's designed to explore, research, optimize, deploy, and initial monitoring all from the same environment (hosted notebook). The IRT item parameters (discriminability, difficulty, guessing, inattention) computed by this pipeline directly supplied the automated curriculum QA dashboards used by the content team; the visualization tooling was built separately and is not included here. On deployment, the IRT-SDT-XGBoost proficiency model reduced student assignment of questions outside the Zone of Proximal Development (65–90% accuracy, measured from live session performance) by *over 35%* compared to the prior staircasing algorithm — saving millions of student-hours previously spent on questions that were too hard or too easy — and outperformed standalone IRT and other traditional psychometric approaches examined during development. Many of the engineering improvements over classical psychometric methods were driven by the realities of the data: practice-mode (non-assessment) question standards, noisy classroom environments, inconsistent labeling, and the general absence of controlled testing conditions that traditional IRT assumes. The accuracy and confidence metrics of this real-time proficiency engine powered a placement project which serves live personalized recommendations.

## Documentation

📚 **[Detailed Technical Documentation](detailed_readme.md)** - Comprehensive guide covering all system components, APIs, and usage examples for engineers.

## System Architecture

🏗️ **[System Architecture Diagram](system_architecture_diagram.md)** - Interactive Mermaid diagram showing component relationships and data flow.

📊 **[High-Level System Schema](SchemaOfML.png)** - Visual overview of the ML container/endpoint architecture and monitoring.

## Core Libraries
*These libraries are required in the directory structure but should not be used directly:*

- **ItemParametersCalculate**: Item Parameter calculator
- **ProficiencyModelTrainingPipeline**: Full end-to-end model training pipeline (data fetch to model artifact export)
- **SnowflakeETL**: Functions for querying and pulling mass data
- **ModelImplementationWSDK/tools/sagemaker_deploy_tools**: Functions for deploying, updating, configuring, etc sagemaker endpoints

## High-Level Workflow
1. Create model artifacts in `WithinSkill_ELA` (or similar project folder) using the above tools
2. Update endpoint in `ModelImplementationWSDK`


## More Detailed Workflow (complete)
1. Create model artifacts in `WithinSkill_ELA` (or similar project folder) by running an automated DS training project to generate a model [e.g. ELA_Proficiency_v6.ipynb]; 
[0. You may want to first run an ItemParameterOnly DS project first to generate a new IP file, or simply use the latest one. NOTE: This can take a very long time to run.]; 
2. Run InferenceCodeTestAndWrite.ipynb to programmatically test model artifacts (model *performance* testing occurs in earlier step, if inference testing flags set to True; detailed results are available in logs), generate inference wrapper script, and export wrapper and artifacts to container.; 
3. Run ModelImplementationWSDK tool within an existing project [if no existing project, use the template ('NewProjectStart.ipynb) to create one and do this BEFORE inference wrapper/artifact export in step 2] to dockerize, test, push to ECR, and deploy on Sagemaker instance (along with testing and rollback if fail). Note: There are some auto-scaling monitoring options at the bottom of the script.
4. Github commit manually, document on Confluence manually, set up/or reference Periscope/Snowflake dashboard to analyze/monitor model performance, manually check Datadog/Cloudwatch for deploy issues, & notify on Slack channel.)

## Project Organization

### Project Folders
- **WithinSkill_ELA**: Project folder for all ELA efforts
- **WithinSkill_Math**: Project folder for all Math efforts

### Model Pipelines
Each project folder contains multiple model version subfolders (e.g., `model_v30`). These versions may have:
- Different data pulls
- Different Item Parameterss
- Different flags
- Various execution options (full pipeline, inference testing only, IP calculation only)

## Step-by-Step Workflow

### 1. Creating a New Pipeline
Run `NewPipelineCopy.ipynb` to create a new pipeline project

### 2. Executing the Pipeline
Use `WithinSkill_ELA/model_v30/ELA_Proficiency_fullRebuild.ipnyb`:

1. Set desired flags (True/False)
2. For a full build:
   - Most parameters should be True
   - Typically keep `ItemParameterCalculate=False` (takes ~3 weeks)
   - IP updates usually run in a separate, asynchronous pipeline
3. For IP recalculation only, set:
   - `ItemParameterCalculate = True`
   - `doSnowflakeETL = True`
   - `pullNewData = True`
   - `GenerateTestTrainSplit = True`

### 3. Model Inference and Testing
Use `InferenceCodeTestAndWrite` to:
- Export model artifacts
- Write an inference wrapper
- Perform local testing of the wrapper
- Execute Flask test of artifacts

Configuration steps:
1. Update artifact directory location
2. Update model version

### 4. AWS Deployment
Use `ModelImplementationWSDK` for AWS Sagemaker endpoint management:
- Do a rolling update of current endpoint
- Change endpoint configuration (autoscaling parameters)
- Run a test of current endpoint
- Monitor scaling

Configuration steps:
1. Update model configuration parameters (rare)
   *Note: First cell configuration details usually don't need modification*
2. Check to see if any unit tests failed
3. Add any new units tests and run them
4. Check if its scaling appropriately
5. Check datadog for API errors with new endpoint

#### For New Endpoints
Use `NewProjectStart` (only for creating brand new endpoints)

### Post-Deployment Documentation
After deployment:
1. Copy model ID and config ID
2. Update live model documentation
3. Push GitHub commit with:
   - Relevant files
   - Artifacts (if size permits)
   - Artifact generation instructions
4. Add GitHub update link to live model documentation: (e.g. https://illuminate.atlassian.net/wiki/spaces/~42953498/pages/17812488211/ELA+Live+Models )
