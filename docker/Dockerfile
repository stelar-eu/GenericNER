FROM python:3.8-slim
WORKDIR /app
COPY ./docker/* /app/
COPY ./src/* /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en_core_web_trf
ENTRYPOINT ["python3", "main.py"]
