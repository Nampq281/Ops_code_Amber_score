
# Pipeline vận hành score
17-Mar-25

Các bước thực hiện: 

Step 0: Chuẩn bị các thư viện
```bash
pip install -r requirements.txt
```

Step 1: parse PCB raw \
Step 2: tính toán feature \
Step 3: gọi model và tính toán ra score
```bash
python single_pipeline.py
```


## Cấu trúc code
```
│   single_pipeline.py  <---File cần chạy     
│   generate_feature.py
│   model_inference.py
│   parse_PCB.py
│   README.md
│   requirements.txt
│
├───artifacts
│       BinProcess_10Mar25.sav     <---Binning process
│       Fiza_PCB_score_10Mar25.sav <---Model
│
├───src
│        config.py
│        f_generator_ops.py
│        utils_ops.py
│        
├───data_input <---Input mẫu dạng pd.DataFrame

```