FROM rasa/rasa
ENTRYPOINT []
CMD []
USER root
RUN apt-get -y update && apt-get -y install \
  build-essential \
  python3-dev

ARG rpath=/rasaGPT
ENV SQLALCHEMY_SILENCE_UBER_WARNING=1 
ENV OPENBLAS_NUM_THREADS=1 
RUN rm -Rf /mnt/www
WORKDIR $rpath
COPY . $rpath
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 5002
EXPOSE 5005
EXPOSE 5055
EXPOSE 8000
