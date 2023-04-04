### 文件结构

```
.
├── README.md
├── data/
├── native-pytorch
│   ├── data_loader.py
│   ├── finetune.py
│   ├── test.py
│   ├── train_from_scratch.py
│   └── utils.py
└── use_trainer
    ├── finetune_use_trainer.py
    ├── my_dataset.py
    └── predict.py
```

* `data`：包含了KdConv与NaturalConv混合的共6个领域的数据集，其中`data.json`的数据信息如下：

|                                | 体育   | 科技  | 教育  | 音乐  | 旅行  | 电影  |
| ------------------------------ | ------ | ----- | ----- | ----- | ----- | ----- |
| # dialogues                    | 9740   | 4061  | 1265  | 1500  | 1500  | 1500  |
| # utterances                   | 195643 | 81587 | 25376 | 24885 | 24093 | 36618 |
| Avg. # utterances per dialogue | 20.1   | 20.1  | 20.1  | 16.6  | 16.1  | 24.4  |

* `native-pytorch`：使用原生PyTorch训练分类器
  * `data_loader.py`：包含dataset与dataloader的实现
  * `utils.py`：包含固定随机种子等工具函数
  * `finetune.py`：使用原生PyTorch微调预训练模型
  * `train_from_scratch.py`：使用Accelerate库从头训练模型
  * `test.py`：测试模型准确率、分辨率等指标
* `use_trainer`：使用🤗Transformers库Trainer训练分类器
  * `my_dataset.py`：包含基于🤗Dataset的dataset
  * `finetune_use_trainer.py`：使用trainer微调预训练模型
  * `predict.py`：使用trainer测试模型准确率、分辨率等指标

