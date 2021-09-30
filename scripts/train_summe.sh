cd ../
python train_avs.py \
  --solver WGAN \
  --summarizer ST-GRAPH-GCN-Diff \
  --n_epochs 50 \
  --split_path  /your/split/file/path/summe_splits.json \
  --dataset_dir  /your/dataset/dir \
  --dataset_name eccv16_dataset_summe_google_pool5 \
  --model_name your_name_for_the_experiment_result \
  --log_dir /your/logs/dir/path \
  --save_dir /your/model/saving/dir/path \
  --score_dir /your/score/dir/path/ \
  --with_images True \
  --video_dir /your/video/dir/path/SumMe/videos \
  --mapping_file /your/mappings/file/path/summe_mapping.json \
  --stgcn_shortcut 1 \
  --variance_loss ssum \
  --sparsity_loss slen \
  --split_ids 0