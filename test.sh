#!/bin/bash

# Base project name
BASE_PROJECT_NAME="IRASim_dino_last_3action_35hor"

# Directory to save wandb links
LOG_DIR="./wandb_logs"
mkdir -p $LOG_DIR

for i in {1..10}
do
    # Generate a unique project name
    PROJECT_NAME="${BASE_PROJECT_NAME}_run_$i"
    
    # Run the Python script with the wandb project name
    python test_in_lt.py --config ./configs/evaluation/languagetable/frame_ada.yaml --wandb_project $PROJECT_NAME
    
    # Save the wandb link
    wandb link=$(wandb run link)
    echo "Run $i: $wandb_link" >> $LOG_DIR/wandb_links.txt
done