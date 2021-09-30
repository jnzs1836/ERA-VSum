# ERA: Entity–Relationship Aware Video Summarization with Wasserstein GAN

## Project Structure


```shell
./cache # cahce for the object detection result
./data # data loaders and video name mapping files
./deployment # code for deployment of the models, e.g. reading the inputting videos.
./evaluation # code for evaluating the results
./factory # factory mode for the solvers and models
./loggers # code for logging the training progress
./notebooks # notebooks for performing the qualitative analysis
./solvers # training solvers based on different settings, i.e. W-GAN and vanilla GAN.
./scripts # scripts for running the training
./models # models used in the project
./utils # utility code for the video summarization
```

## Installation
```shell
pip install requirements.txt
```
If you encounter the errors regarding Detectron2, please check the [document](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).


## Running
The entrypoint of our project is the file train_avs.py. We also provide two bash scripts in scripts directory.
```shell
bash ./scripts/train_tvsum.sh # train models on TVSum

bash ./scripts/train_summe.sh # train models on SumMe
```

## Evaluation
We offer a trained model checkpoint in the chcekpoints directory. You could test the model on your own dataset and splits.
The model is trained on the SumMe split-3. Due to the file size limit, we are only able to add one checkpoint file in the submission.
```shell
python generate_scores.py \
	--ckpt_path /your/checkpoint/dir/split-x.pkl \
	--model_name custom_name_for_saving_the_result \
    --output_dir /your/output/dir \
    --split_index 0
```

## Acknowledgement
We thank to [j-min](https://github.com/j-min/Adversarial_Video_Summary) for providing the implementation of the original SUM-GAN.
