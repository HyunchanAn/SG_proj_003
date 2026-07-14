FROM python:3.10-slim

RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app



COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

RUN python3.10 -m pip install httpx requests pydantic-settings pandas numpy scikit-learn rdkit torch sqlalchemy
