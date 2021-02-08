# BERT(S) for Relation Extraction

### Original README

https://github.com/plkmo/BERT-Relation-Extraction

### 2021-02-02

- 加入了中文的支持：`model_no=3` 即为使用本地的中文 bert，默认位置是 `/home/diske/ivenwang/data/prev_trained_model/bert-base`，以后要改成参数
- 加入了新任务：`task=elec` 即为使用电力数据集
- 对于不平衡的数据：复制 50 遍，参数在 `processing_funcs.preprocess_elec` 中
- 新增数据后，训练模型时别忘记修改 `--num_classes` 参数！

### 2021-02-08

- 