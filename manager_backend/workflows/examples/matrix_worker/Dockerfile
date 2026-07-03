FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir numpy==2.1.3
COPY matrix_add.py matrix_multiply.py ./

# Le volontaire monte /input et /output
ENTRYPOINT ["python"]
CMD ["matrix_add.py"]
