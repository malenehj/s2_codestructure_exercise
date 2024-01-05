# Base image
FROM --platform=linux/amd64 python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY s2_codestructure_exercise/ s2_codestructure_exercise/
COPY data/ data/
COPY models/ models/
COPY Makefile Makefile

WORKDIR /
RUN pip install . --no-cache-dir #(1)

ENTRYPOINT ["python", "-u", "s2_codestructure_exercise/train_model.py"]