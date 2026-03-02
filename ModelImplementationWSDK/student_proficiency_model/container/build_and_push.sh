# This script isn't used currently, so we can dynmically label IMAGE_TAG in our python
# entire script replicated in the docker build part of the BuildAndDeployDockerImageToSagemakerEndpoint.ipynb
#!/bin/bash

# Set variables
AWS_ACCOUNT_ID="[REDACTED]"
REGION="us-east-1"
ECR_REPOSITORY="custom-logistic-model"
IMAGE_TAG="latest"

# Authenticate Docker to AWS ECR (this will work for all repositories in the specified region)
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$REGION.amazonaws.com

# Also authenticate to your own ECR repository
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} || aws ecr create-repository --repository-name ${ECR_REPOSITORY}

# Build the Docker image
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

# Tag the image for ECR
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Push the image to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

echo "Image pushed successfully to ECR: $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"