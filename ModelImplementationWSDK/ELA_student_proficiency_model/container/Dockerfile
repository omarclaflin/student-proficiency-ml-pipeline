# Start with the AWS Deep Learning Container for Python 3.8
#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.1-cpu-py38-ubuntu20.04
# FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04-sagemaker

# Set working directory
WORKDIR /opt/ml/model

# Install the required packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    flask \
    gunicorn \
    pickle5 
RUN pip install --no-cache-dir -v xgboost==2.1.1
RUN pip install --no-cache-dir joblib==1.4.2
RUN pip install --no-cache-dir scikit-learn
#RUN pip install --no-cache-dir numpy==1.26.4
#RUN pip install --no-cache-dir pandas==1.5.3

# Copy your model file and any other necessary files
COPY proficiency_model.json /opt/ml/model/
#COPY proficiency_scaler.json /opt/ml/model/
COPY confidence_score_scaler.json /opt/ml/model/
COPY confidence_model.json /opt/ml/model/
COPY item_params.csv /opt/ml/model/
COPY inference.py /opt/ml/model/


# Set the entrypoint to gunicorn
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "inference:app"]