FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
EXPOSE 8000
CMD ["python", "api.py"]

