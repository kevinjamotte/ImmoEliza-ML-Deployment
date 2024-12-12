# Use the official Python image from Docker Hub as a base image
FROM python:3.13

# Set the working directory inside the container
WORKDIR /app

# Install distutils (which is required by setuptools)
RUN apt-get update && apt-get install -y python3-distutils

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Set the command to run your Streamlit app
CMD ["streamlit", "run", "app.py"]
