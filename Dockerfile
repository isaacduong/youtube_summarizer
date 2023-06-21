# Use the official Python base image
FROM python:3

# Set the working directory inside the container
WORKDIR /app

# Install Git
RUN apt-get update && apt-get install -y git

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code to the working directory
COPY . .

# Specify the command to run when the container starts
#CMD ["python", "yt_summarizer/youtube_video_summary.py"]

