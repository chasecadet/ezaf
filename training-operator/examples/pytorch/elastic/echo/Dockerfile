FROM python:3.8-buster
WORKDIR /workspace
RUN pip install torch==1.13.1 numpy
# TODO Replace this with the PIP version when available
ADD examples/pytorch/elastic/echo/echo.py echo.py
ENV PYTHONPATH /workspace
ENV ALLOW_NONE_AUTHENTICATION yes
ENTRYPOINT ["python", "-m", "torch.distributed.run"]
