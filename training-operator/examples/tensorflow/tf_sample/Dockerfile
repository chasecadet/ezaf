FROM  tensorflow/tensorflow:1.8.0
RUN pip install retrying
RUN mkdir -p /opt/kubeflow
COPY examples/tensorflow/tf_sample/tf_smoke.py /opt/kubeflow/
ENTRYPOINT ["python", "/opt/kubeflow/tf_smoke.py"]
