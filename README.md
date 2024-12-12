# Zero-Shot VC

环境安装

```shell
pip install -r requirments.txt

```

模型下载

```shell
modelscope download --model 'ACoderPassBy/UnetVC' --local_dir './files'

```

快速使用

```shell
python converter.py --source_wav /path/source.wav --target_wav /path/target.wav --save_path /path/converted.wav --model_path ./files
```


# 感谢

FunASR

