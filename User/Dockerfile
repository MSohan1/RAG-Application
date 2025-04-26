FROM python:3.11
EXPOSE 8084
WORKDIR /app
COPY requiments.txt ./
RUN pip install -r requiments.txt
COPY . ./
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8084", "--server.address=0.0.0.0" ]