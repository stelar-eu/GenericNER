FROM python:3.8-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    curl \
    jq \
 && rm -rf /var/lib/apt/lists/*
COPY ./docker/* /app/
COPY ./src/* /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_trf
ENTRYPOINT ["./run.sh"]
