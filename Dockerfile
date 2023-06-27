FROM python:3.10-buster
WORKDIR ./app
COPY requirements.txt .
RUN pip install git+https://github.com/patrickjohncyh/fashion-clip.git
RUN pip install -r requirements.txt
RUN pip install urllib3==1.25
COPY . .
CMD ["python", "app.py"]