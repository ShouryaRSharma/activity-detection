ARG POETRY_VERSION=1.4.2

FROM python:3.11.4 AS base

ARG POETRY_VERSION

RUN pip install poetry==${POETRY_VERSION} --no-cache-dir

WORKDIR /temp

COPY ./pyproject.toml /temp/pyproject.toml
COPY ./poetry.lock /temp/poetry.lock

RUN poetry config virtualenvs.create false \
    && poetry export --without-hashes -f requirements.txt --output requirements.txt \
    && poetry export --without-hashes --dev -f requirements.txt --output requirements-dev.txt

FROM python:3.11.4-slim-buster AS app-base

ARG POETRY_VERSION

WORKDIR /code

RUN apt-get update && apt-get install -y curl \
    && pip install poetry==${POETRY_VERSION} --no-cache-dir

RUN addgroup --system activity_detection \
    && adduser --system --ingroup activity_detection activity_detection \
    && chown -R activity_detection:activity_detection /usr/local/lib/python3.11/site-packages \
    && chown -R activity_detection:activity_detection /usr/local/bin \
    && chown -R activity_detection:activity_detection /code

FROM app-base AS test

COPY --from=base /temp/requirements-dev.txt /code/requirements-dev.txt

# Installing dev requirements as root
RUN pip install --upgrade -r requirements-dev.txt --no-cache-dir

USER activity_detection

COPY ./activity_detection /code/activity_detection
COPY ./tests /code/tests
COPY ./pyproject.toml /code/pyproject.toml
COPY ./poetry.lock /code/poetry.lock

ENTRYPOINT ["python", "-m", "activity_detection.main"]

FROM app-base AS production

COPY --from=base /temp/requirements.txt /code/requirements.txt

RUN pip install --upgrade -r requirements.txt --no-cache-dir

USER activity_detection

COPY ./activity_detection /code/activity_detection
COPY ./pyproject.toml /code/pyproject.toml
COPY ./poetry.lock /code/poetry.lock

ENTRYPOINT ["python", "-m", "activity_detection.main"]
