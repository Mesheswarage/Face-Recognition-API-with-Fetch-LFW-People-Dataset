# Image-Classification-API-with-Fetch-LFW-People-Dataset
This project features a custom face recognition model trained on the Fetch LFW People dataset using scikit-learn and it provides easy integration through a simple API.

## Input
![Screenshot (600)](https://github.com/Mesheswarage/Image-Classification-API-with-Fetch-LFW-People-Dataset/assets/97176530/1618c790-c1d5-4b86-aa34-d10a8defa032)

## Output

![Screenshot (601)](https://github.com/Mesheswarage/Image-Classification-API-with-Fetch-LFW-People-Dataset/assets/97176530/485f7f20-df70-456d-b15e-0c5593653f9d)

This API using FastAPI for face recognition

`uvicorn` is used to run the FastAPI app, `pickle` for loading the trained model and PCA instance, and `PCA` from scikit-learn for Principal Component Analysis

API requests, expects a file (image) to be uploaded as part of the POST request.

Inside the request handler, the code performs the following steps:


Reads the uploaded image as bytes.

Converts the image to grayscale.

Converts the image to a NumPy array and scales its values to the range [0, 1].

Applies the fitted PCA transformation to the preprocessed image data.

Uses the trained machine learning model to make predictions based on the transformed input data. The result is assumed to be a single class index.

Returns the predicted class as a JSON response with a 200 status code.

If an exception occurs during image processing or prediction, it returns an error response with a 500 status code.
