# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Install gfortran and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    gfortran \
    gcc \
    libblas-dev \
    liblapack-dev

# Set the working directory to /app
WORKDIR /starter

# Add your application to the PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Getting aws credentials from secret files
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

# Copy the current directory contents into the container at /app
COPY . /starter

# Install any needed packages specified in requirements.txt
RUN pip install dvc \ 
    dvc[s3] \
    flake8 pytest \
    fastapi==0.95.1 \
    imbalanced_learn==0.7.0 \
    joblib==1.2.0 \
    numpy==1.24.2 \
    pandas==2.0.0 \
    pydantic==1.10.7 \
    pytest==7.3.1 \
    scikit_learn==1.2.2 \
    scipy==1.10.1 \
    setuptools==67.6.1 \
    xgboost==1.7.4 \
    httpx==0.24.0 \
    uvicorn==0.16.0

# Make port 80 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PORT 8000

CMD flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics ; \
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics ; \
dvc init ; \
dvc remote add -d s3_remote s3://fastapi-ds-project ; \
dvc remote modify s3_remote region us-east-1 ; \
dvc remote modify s3_remote access_key_id ${AWS_ACCESS_KEY_ID} ; \
dvc remote modify s3_remote secret_access_key ${AWS_SECRET_ACCESS_KEY} ; \
dvc pull -v; \
pytest ; \
cd starter ; \
uvicorn main:app