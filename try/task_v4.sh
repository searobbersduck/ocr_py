python ocr_model_v2_try.py --time_steps 240 --iter_num 40000 --batch_size 384
mv ./models ./models-seq240
python ocr_model_v2_try.py --time_steps 480 --iter_num 45000 --batch_size 256
mv ./models ./models-seq480