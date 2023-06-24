ARG BASE_IMAGE=rasa/rasa
FROM ${BASE_IMAGE} as hg-base
USER root
RUN apt-get -y update && apt-get -y install \
  vim 

FROM hg-base as hg
ARG hgpath=/hg
EXPOSE 8000
EXPOSE 5055
RUN useradd -m -s /bin/bash -d /home/python python
RUN rm -Rf /mnt/www
WORKDIR $hgpath
COPY ../. $hgpath
RUN pip install --upgrade pip
RUN pip install -r ../requirements.txt --no-cache-dir
 

FROM hg
ARG hgpath=/hg
WORKDIR $hgpath
ENV SQLALCHEMY_SILENCE_UBER_WARNING=1 
USER python
CMD ['pwd']
#CMD ['rasa' 'run' '--cors' '"*"']
