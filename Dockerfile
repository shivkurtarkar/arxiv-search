FROM python:3.8-slim-buster AS ApiImage

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app/
COPY ./data/ ./data

RUN mkdir -p /app/backend
WORKDIR /app/backend

COPY ./backend/ .
RUN pip install -e .

WORKDIR /app/backend/vecsim_app

CMD ["sh", "./entrypoint.sh"]