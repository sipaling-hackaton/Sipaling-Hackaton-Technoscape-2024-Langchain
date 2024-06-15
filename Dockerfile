FROM python:3.9-slim

WORKDIR /opt

COPY requirements.txt /opt/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /opt/requirements.txt

COPY . /opt

CMD ["fastapi", "run", "main.py", "--port", "8080", "--host", "0.0.0.0"]