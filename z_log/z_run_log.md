examples-main: pytorch exp

手动下载数据集： cifar-10, mnist

# 在pycharm ide 的环境中跑 example/minst/main: ide会自动识别 test() 函数名导致运行环境为测试环境，入参model为None报错， 需要
# 手动编辑配置文件：指定 python main 文件及具体路径
# 或手动修改 test函数名(鉴于方便调试)

# #################################

example-kaggle:
tabular data: 
    cibmtr-queit_in_post_HCT:  EDA 表格类数据处理：分类问题处理模板：template

cv data: 
    构造网络模型 处理图像识别类问题 

# ##################################

examples: qlib exp 
手动下载数据集：qlib-stock-cn
cd scripts && python get_data.py qlib_data --target_dir ~/.qlib_bak/qlib_data/cn_data --region cn
依赖包路径： qlib/tests/data-> 113行打断点 debug scripts/get_data 配置脚本执行参数(qlib_data --target_dir ~/.qlib_bak/qlib_data/cn_data --region cn)找到下载路径
https://github.com/SunsetWolf/qlib_dataset/releases/download/v2/qlib_data_cn_1d_0.9.zip
https://github.com/SunsetWolf/qlib_dataset/releases/tag/v3 注意版本更新 v3 

# #################################
pip freeze > requirements.txt