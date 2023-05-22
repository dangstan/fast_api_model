# README
This repository contains the codebase for training an XGBoost model and using the model to make predictions via an API. The model is trained on cleaned data from the census.csv file. The project also includes a robust testing suite and is designed for Docker-based deployment on the Render web service. GitHub Actions are used for continuous integration, ensuring that testing is automatically run and passed before changes are deployed.

# Project Setup

### Environment Setup

1. Download and install Docker if you don’t have it already.
2. Clone this repository.
3. Run `docker build -t [image_name]` . to build the Docker image for this project.
4. Run `docker run -d -p 8000:8000 [image_name]` to start the Docker container. The API should now be accessible at `localhost:8000`.


### Repository Usage
1. Clone the repository and navigate to the project's root directory.
2. Make code changes as needed, and frequently commit your changes.


### Continuous Integration
The repository is configured with GitHub Actions for continuous integration. Upon each push to the repository, the suite of unit tests and flake8 linting checks are automatically run. Commits will only be accepted if both tests and linting pass without error.

### Data
The census.csv file contains the raw data for this project. This data is preprocessed using scripts found in `ml/data.py`. After cleaning, the processed data is used to train the machine learning model.

### Model
The machine learning model is a grid-search optimized XGBoost model trained using KFold cross-validation and BorderlineSMOTE for oversampling. The model is trained using the script found in `ml/train_model.py`, and the trained model is saved as a serialized `.pkl` file for later use.

Performance of the model on data slices (particularly for categorical features) is also assessed and outputted for review. This functionality can be found within `ml/model.py`.

Unit tests for various functions in the model code can be found under the `starter` directory.

### API
A RESTful API is provided for interfacing with the trained model. This API is built using FastAPI and offers the following endpoints:

1. `GET /` : Returns a welcome message.
2. `POST /predict` : Accepts input data and returns a prediction from the trained model.
The API script can be found in `model/predict.py`. This script loads the trained model and defines the API endpoints.

API unit tests are included in the `starter` directory, providing coverage for both `GET` and `POST` requests.

### Deployment
Deployment is handled via Docker and the Render web service. A Dockerfile is provided for building a Docker image of the API server.

For deployment on Render:

1. Create a Render account and follow the instructions for deploying a new Docker-based service.
2. Point the new service to your Docker image.

After successful deployment, the API can be accessed at the provided Render service URL.