#!/bin/bash

# The name for your model
algorithm_name=chatterbox-streaming-sagemaker

# Get the AWS account number
account=$(aws sts get-caller-identity --query Account --output text)

# Get the AWS region
region=$(aws configure get region)
region=${region:-us-east-1}

# The full name of the ECR repository
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

# If the repository doesn't exist, create it
aws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

# Get the login command from ECR and execute it
aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"

# Build the Docker image and push it to ECR
docker build -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}