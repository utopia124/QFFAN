# 基础镜像
FROM pytorch
# 维护者信息
LABEL maintainer ="liul29956@gmail.com"
# 工作路径
WORKDIR /QFFAN
# 构建上下文目录
ADD ../../QFFAN .
# 安装第三方依赖
RUN  pip install requirements.txt
# 运行代码
CMD ["python", "hello.py"]







