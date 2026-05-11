FROM python:3.11

WORKDIR /code

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces expose port 7860
EXPOSE 7860

# Run the Flask app
CMD ["python", "flask_app.py"]
