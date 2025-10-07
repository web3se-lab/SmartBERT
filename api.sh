export CUDA_VISIBLE_DEVICES=0 # set cuda if you have, choose the GPUs
export API_MODEL=./model/SmartBERT-v2 # set model, choose a SmartBERT model to load, default is SmartBERT-V2-codebert
export API_PORT=9900 # set web API port, default is 9900

python3 ./api.py
