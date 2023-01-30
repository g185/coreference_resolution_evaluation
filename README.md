# Coreference Resolution Evaluation 

usage:
```
python eval.py 
        --model MODELNAME{lingmess, fastcoref} (default:lingmess)
        --custom_test_path CUSTOM_TEST_PATH  (default:inputs/test.english.jsonlines)
        --custom_output_path (default:outputs/output)
        --cpu (else evaluated on cuda:0)
```