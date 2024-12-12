export CUDA_VISIBLE_DEVICES=1 # set cuda, choose the GPUs
export API_MODEL=SmartBERT-V3 # set model, choose a SmartBERT model to load, default is SmartBERT-V2-codebert
export API_PORT=9100 # set web API port, default is 9100

python3 ./api.py
