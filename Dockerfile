# --- Stage 1: Use an official Python runtime as a parent image ---
# Using python:3.11-slim for a good balance of features and size.
FROM python:3.11-slim

# --- Stage 2: Set up the environment ---
# Set the working directory inside the container
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- Stage 3: Install dependencies ---
# First, copy only the requirements file to leverage Docker's layer caching.
# This step is only re-run if requirements.txt changes.
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# --- Stage 4: Copy the application code and models ---
# Copy the application code, trained models, and templates into the container
COPY ./app /app/app
COPY ./models /app/models

# --- Stage 5: Expose port and run the application ---
# Expose the port the app runs on
EXPOSE 8000

# The command to run your app using uvicorn.
# Host 0.0.0.0 makes it accessible from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]