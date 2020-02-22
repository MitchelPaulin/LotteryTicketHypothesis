export ZONE="us-east1-b"
export INSTANCE_NAME="my-fastai-instance"
gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080