## nnに関する考察
- 層を深くすればするほど精度が下がる
- denseを1/2させていくとスコアは正常になる
  
### optimaizer
adaDeltaではかなり精度が下がりそう...  
なんでこのカーネルはこんなにいい精度が出ているのか...  
https://www.kaggle.com/diegosiebra/neural-network-model-for-house-prices-keras



## 主成分分析で200絡むを60カラムに圧縮
該当ファイル：train_3
xgboostで0.13248から0.78602になった  
valiのスコアは悪くない...  
適切に標準化して60にしたら0.16327  