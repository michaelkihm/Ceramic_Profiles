FROM python:3.7

RUN apt update -y && apt install build-essential -y &&  apt-get install manpages-dev -y
RUN pip install --upgrade pip
RUN pip install opencv-python && \
    pip install numpy && \
    pip install scikit-learn && \
    pip install pandas && \
    pip install pylint && \
    pip install jupyter && \
    pip install matplotlib && \
    pip install imutils


ENTRYPOINT ["jupyter", "notebook", "--ip='0.0.0.0'", "--no-browser", "--allow-root"]
