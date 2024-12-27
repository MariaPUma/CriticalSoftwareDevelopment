# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /practica3

# Copy the current directory contents into the container at /app
ADD . /practica3

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# # Make port 80 available to the world outside this container
EXPOSE 80

# Run practica3.py when the container launches
CMD ["python", "practica3.py"]
