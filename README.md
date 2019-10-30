# HQFSVM-SA
* 对dna和protein序列进行分类，输入fasta文件，程序自动识别序列类型，并根据相应的打分矩阵进行计算。
* 对于dna序列 match ：1 ， mismatch ： -3 ，  gap open ：-5 ， gap extend ：-2 ，
* 对于protein序列采用的是BLOSUM62打分矩，gap open : -10 , gap extend ： -0.5 。

### 环境
***********
* jdk 1.8
* libsvm
* weka >=3.8

### 使用方法（在命令行下测试时，使用和HQFSVM-SA.zip同级目录的HQFSVM-SA.jar。）
***********************

'''
* 输入文件：arff格式
* 辅助程序：Fasta2arff.py &emsp; python &emsp; Fasta2arff.py &emsp; -f &emsp; xx.fasta &emsp; -a&emsp;  xx.arff &emsp;   -l &emsp; 正例数
* 交叉验证：java &emsp;-jar &emsp; HQFSVM-SA.jar  &emsp; -f  &emsp; trainfile &emsp;  -c &emsp; cv
* 独立测试：java &emsp;-jar &emsp; HQFSVM-SA.jar  &emsp; -f  &emsp; trainfile &emsp;  -p &emsp; testfile &emsp; 
* 测试用例：见example文件夹
'''

### Weka 安装教程

******************************
* 将 HQFSVM-SA.zip 按照如下教程安装。
* 参考地址：[加载自定义分类器到weka](https://blog.csdn.net/So_that/article/details/82915198)
