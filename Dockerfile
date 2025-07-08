FROM python:3.9-slim
WORKDIR /opt/program

# Install git for pip install from git repos
RUN apt-get update && apt-get install -y git

# Copy requirements and install dependencies
# pip will now use git to clone and install chatterbox-streaming
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ /opt/program/app

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM app.app:app

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.app:app"]