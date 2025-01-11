一、硬件配置：
**CPU:Intel Core i9-14900HX  24 cores**
**GPU:RTX 4070 Laptop 8GB**
**RAM:32GB**

训练参考时间：Qwen2.5-0.5B-Instruct 、1万多token、10个epoch，Lora训练模式约31个小时（还是得有钱买好的卡才好啊=-=）

二、基座模型和数据集

BaseModel：Qwen2.5-0.5B-Instruct

https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files

Dataset：IndustryInstruction医疗健康SFT数据集
https://www.modelscope.cn/datasets/BAAI/IndustryInstruction_Health-Medicine/files

三、目录/文件说明

1、FT_train.jsonl：模型支持的数据集格式之一，自定义数据集需要按照此格式进行处理

2、目录datasets：其中为从上述数据集中下载的预览数据集（只是学习用，没下载完全，完整数据集请自行去上述网址下载）：

medical_train_datasets.csv

medical_test_datasets.csv

目录下的其他数据集为学习过程自定义的一些文件，不用管

3、将数据集转换成jsonl格式

FT_get_train_dataset_jsonl.py：处理训练数据集，得到  train_datasets_generated.jsonl

FT_get_val_dataset_jsonl.py：处理验证数据集，得到 val_datasets_generated.jsonl

4、微调

FT_train.sh：其中可以定义微调的方式（full或者lora等），也可以自定义train和val的数据集，定义GPU等，详情请参考Swift官网：[ swift 3.1.0.dev0 文档](https://swift.readthedocs.io/zh-cn/latest/GetStarted/快速开始.html)
merge_lora.sh：合并lora权重和base model权重

5、推理

FT_infer_lora.py：如果微调方式为Lora，参考此推理代码进行推理

FT_infer_full.py：如果微调方式为full，参考此推理代码进行推理


其他推理方式，请参考qwen2.5的详细说明



