FROM python:3.11

WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY . .


EXPOSE 7860

# Run the Flask app
CMD ["python", "flask_app.py"]
