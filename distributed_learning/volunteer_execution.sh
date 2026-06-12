docker compose up -d

docker exec -it distributed_learning-volunteer-1 bash

export $(grep -v '^#' .env | xargs)

python3 volunteer.py