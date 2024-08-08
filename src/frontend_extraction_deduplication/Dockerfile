# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 9072

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=9072", "--server.enableCORS=false"]