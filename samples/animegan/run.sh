compose_file=$(dirname $0)/docker-compose.x86.yml
module_container_name=$(basename $(dirname $0))-module-1

docker compose -f $compose_file up -d
echo "Waiting for $module_container_name to finish"
docker container wait $module_container_name
docker compose -f $compose_file down
