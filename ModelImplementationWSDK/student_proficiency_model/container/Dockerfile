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
    scikit-learn==1.3.2 \
    pandas \
    flask \
    gunicorn \
    pickle5

# Copy your model file and any other necessary files
COPY model.pkl /opt/ml/model/
COPY inference.py /opt/ml/model/

# Set the entrypoint to gunicorn
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "inference:app"]