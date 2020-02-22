# to use run ./uploadFiles.sh <path to file>

export INSTANCE_NAME="my-fastai-instance"
gcloud compute scp $1 $INSTANCE_NAME:~/uploads
