FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY . /app/playwright_crawling
WORKDIR /app/playwright_crawling

RUN uv venv /opt/venv
# Use the virtual environment automatically
ENV VIRTUAL_ENV=/opt/venv

ENV PLAYWRIGHT_BROWSERS_PATH=/playwright-browsers

RUN uv pip install --system \
    --no-cache -r requirements.txt



RUN pip install playwright \
    && playwright install --with-deps chromium \
    && chmod -Rf 777 $PLAYWRIGHT_BROWSERS_PATH

