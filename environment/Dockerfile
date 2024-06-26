# hash:sha256:e0eff0151f1fa72a20d2073a0368a786976f74a9f89ac9f97cef1ebc747ccaf4
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=pytorch --channel=conda-forge \
        cudatoolkit==10.1.243 \
        ifcfg==0.21 \
        matplotlib==3.3.2 \
        opencv==3.4.2 \
        pytest-socket==0.3.5 \
        pytorch==1.7.0 \
        scipy==1.5.3 \
        tensorboard==2.4.0 \
        torchvision==0.8.1 \
        tqdm==4.55.2 \
    && conda clean -ya

COPY postInstall /
RUN /postInstall
