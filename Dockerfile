# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Install gfortran and other dependencies
RUN apt-get update && apt-get install -y \
    gfortran \
    gcc \
    libblas-dev \
    liblapack-dev

# Set the working directory to /app
WORKDIR /starter

# Add your application to the PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Copy the current directory contents into the container at /app
COPY . /starter

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install dvc dvc[s3] flake8 pytest

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV PORT 80

CMD flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics && \
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics \
pytest \
uvicorn starter.main:app --host 0.0.0.0 --port $PORT