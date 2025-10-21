FROM python:3.11-slim

# system packages needed by some libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl cmake libopenblas-dev liblapack-dev libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements and install first (caches layers)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# copy app
COPY . /app

# ensure upload folder exists (if used)
RUN mkdir -p /app/uploads

EXPOSE 10000
CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 3 --threads 4
