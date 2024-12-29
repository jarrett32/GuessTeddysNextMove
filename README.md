# Guess Teddys Next Move

This is a project to train a CV model on my brothers dog Teddy. Teddy is a wild boy and no one can guess his best move so I decided to put some ML models to the test!


## Table of Contents
- [Guess Teddys Next Move](#guess-teddys-next-move)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
  - [Data Collection](#data-collection)
  - [Prediction Machine Learning Model](#prediction-machine-learning-model)

## Setup

- Setup your AWS credentials by creating a `.env` file with the following contents:
- Setup `.env` and `.terraform.tfvars` files using the example files
- Then run the following commands:
  
```bash
pip install -r requirements.txt
terraform init
terraform plan
terraform apply
```

## Data Collection

I have two blink cameras mini from amazon, theyre super cheap and I got a lightning deal for around $10 each. While these cameras have some technical limitations regarding video recording capabilities (without paying) and developer API access, theyre probably not the best for cv apps. But theyre good enough for this project.

First you need to download the app, set up any number of blink cameras. Add you username and password to the `.env` file. Then run the following command to start taking pictures each second for every camera:
```bash
python src/data_collection/main.py
```

To upload them to s3 create a folder `data/saved` and run 
```bash
python src/data_collection/uploadToS3.py
```


*You can easily seperate the buckets by changing the `data_collection_bucket_name` in you tfvars file rerunning the process.*


## Prediction Machine Learning Model
`pip uninstall -y jax jaxlib` if m1 mac is giving you problems