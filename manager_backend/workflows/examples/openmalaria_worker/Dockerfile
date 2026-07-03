FROM python:3.11-slim

WORKDIR /app
COPY run_simulation.py .

# Le volontaire monte /input (scenario.xml) et /output
ENTRYPOINT ["python", "run_simulation.py"]
