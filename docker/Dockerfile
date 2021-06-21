FROM informaticsmatters/rdkit-python3-debian:Release_2019_03_1
USER 0
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN apt-get update && apt-get install -y \
  graphviz \
  xdg-utils &&\
  apt-get clean && \ 
  rm -rf /var/lib/apt/lists/* &&\
  # pip --no-cache-dir install --upgrade pip && \
  pip --no-cache-dir install -r requirements.txt
COPY . /code/