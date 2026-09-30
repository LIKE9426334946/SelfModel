# ModelTrial


# 注意事项
requirements.txt现在只适用于CPU版本，不适用于GPU版本  


# 使用虚拟环境
.\venv\Scripts\Activate.ps1  

# 运行
如果自己指定数据集根目录，用--root指定，指定模型，用--model指定 
python -m train --model "MODELNAME" --root ".../datasets"

# 测试，和运行方法一样
