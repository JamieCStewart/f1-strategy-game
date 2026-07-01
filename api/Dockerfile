FROM python:3.12-slim
WORKDIR /app
COPY sim/ ./sim/
RUN pip install --no-cache-dir -e ./sim
COPY api/ ./api/
RUN pip install --no-cache-dir -e ./api
CMD ["pitwall-server"]
