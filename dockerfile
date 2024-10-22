FROM python:3.11

RUN pip install torch==2.4.1 transformers==4.44.2 peft==0.13.2 trl==0.11.4 

RUN pip install fastapi uvicorn 

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
