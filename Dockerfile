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
RUN pip install --no-cache-dir -r requirements.txt && pip install dvc dvc[s3]

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV PORT 80

# Run app.py when the container launches
CMD uvicorn starter.main:app --host 0.0.0.0 --port $PORT