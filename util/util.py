import jsonlines
import subprocess


#fastcoref cluster correction (x[1]-1)
def correct_fastcoref_clusters(list_of_clusters):
    corrected_list_of_clusters = []
    for cluster in list_of_clusters:
        correct = [[c[0], c[1]-1] for c in cluster]
        corrected_list_of_clusters.append(correct)
    return corrected_list_of_clusters

def write_docs_to_jsonlines(path, name, docs):
    #write
    with jsonlines.open(path + name + ".jsonlines", mode="w") as f:
        for line in docs:
            f.write(line)

def jsonlines_to_html(jsonlines_input_name):
    subprocess.call("python3 util/corefconversion/jsonlines2text.py {}.jsonlines -i -o {}.html --sing-color"
                   " \"\" --cm \"common\"".format(jsonlines_input_name, jsonlines_input_name), shell=True)

def jsonlines_to_conll(jsonlines_input_name):
    subprocess.check_call("python3 util/corefconversion/jsonlines2conll.py -g {}.jsonlines "
                   "-o {}.conll".format(jsonlines_input_name, jsonlines_input_name), shell=True)


#def merge_conll_jsonlines(jsonlines_input_name, conll_ref_name, output_directory_path):
#    subprocess.check_call("python3 util/corefconversion/jsonlines2conll.py -g {}.jsonlines "
#                   "-o {}merged.conll -c {}.conll".format(jsonlines_input_name, output_directory_path, conll_ref_name), shell=True)


