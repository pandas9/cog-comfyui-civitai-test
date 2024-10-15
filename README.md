# Readme

run ./scripts/install_custom_nodes.py to install the custom nodes (or ./scripts/reset.py to reinstall ComfyUI and all custom nodes)

download weights before pushing the model

df -h

sudo du -ahx / | sort -rh | head -n 20

Stop all running Docker containers:
docker stop $(docker ps -aq)

Remove all Docker images:
docker rmi $(docker images -q)

If you have no images, you'll see a similar error, which is fine.
Remove all Docker volumes:
docker volume rm $(docker volume ls -q)

Remove all Docker networks (except default ones):
docker network prune -f

Remove the Docker build cache:
docker builder prune -af

For a comprehensive cleanup:
docker system prune -af --volumes

docker system prune -a

docker system prune -af --volumes

completely reset Docker and remove all of its data
sudo systemctl stop docker
sudo rm -rf /var/lib/docker

sudo systemctl start docker
