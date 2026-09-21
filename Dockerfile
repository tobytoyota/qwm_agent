FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get -y install gcc g++
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
RUN conda install -y python=3.12
RUN conda install -y scipy numpy matplotlib numba
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding pyaml igraph ujson
RUN conda install pandas
RUN pip install networkx dill pyastar2d easydict sympy pygame
COPY . .
CMD ["/bin/bash"]
