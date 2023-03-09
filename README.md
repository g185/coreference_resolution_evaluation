# Coreference Resolution Evaluation 

usage:
```
python eval.py 
        --model MODELNAME {lingmess, fastcoref} (required: False)
        --custom_input_path CUSTOM_INPUT_PATH (required: False)
        --custom_test_path CUSTOM_TEST_PATH  (default:test.english.jsonlines)
        --custom_output_path (default:output)
        --cpu (else evaluated on cuda:0)
```