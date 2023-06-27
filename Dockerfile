FROM python:3.10-buster
WORKDIR ./app
COPY requirements.txt .
RUN pip install git+https://github.com/patrickjohncyh/fashion-clip.git
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]