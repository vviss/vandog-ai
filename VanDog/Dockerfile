FROM python:3.7

RUN set -ex && mkdir /translator
WORKDIR /VanDog

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git

COPY . ./

EXPOSE 5000
ENV PYTHONPATH /VanDog
CMD python3 /VanDog/app.py