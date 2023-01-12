from fastcoref import LingMessCoref, FCoref
from util.util import *
import argparse

from metrics import MentionEvaluator, CoNLL2012CorefEvaluator

#returns: docs: dict("sentences":[["w1","w2",...]], "clusters":[[[wstart,wend]...]]
def extract_and_preprocess(input:str = "test.english.jsonlines"):
    docs = []
    with jsonlines.open(input) as f:
        for elem in f:
            if "speakers" in elem: del elem["speakers"]
            if "constituents" in elem: del elem["constituents"]
            if "ner" in elem: del elem["ner"]
            docs.append(elem)
    return docs

#returns: docs with predicted clusters
def predict(model, docs):
    pred = []
    for doc in docs:
        elem = doc.copy()
        list_of_clusters = model.predict([item for sublist in doc["sentences"] for item in sublist], is_split_into_words=True).get_clusters(as_strings=False)
        elem["clusters"] = correct_fastcoref_clusters(list_of_clusters)
        pred.append(elem)
    return pred

def evaluate_mention_scores(pred_mentions, gold_mentions):
    evaluator = MentionEvaluator()

    for pred, gold in zip(pred_mentions, gold_mentions):
        evaluator.update(pred, gold)

    precision, recall, f1_score = evaluator.get_prf()
    return precision, recall, f1_score

def evaluate_coref_scores(pred, gold, mention_to_pred, mention_to_gold):
    evaluator = CoNLL2012CorefEvaluator()

    for p, g, m2p, m2g in zip(pred, gold, mention_to_pred, mention_to_gold):
        evaluator.update(p, g, m2p, m2g)
    result = []
    for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
        result.append(dict(zip(["precision", "recall", "f1_score"], evaluator.get_prf(metric))))
    return result

#outputs all coreference scores
def write_scores_report(pred, gold, pred_output_name, gold_output_name):
    gold_mentions, gold_clusters, gold_mention_to_clusters = extract_from_results(gold)
    pred_mentions, pred_clusters, pred_mention_to_clusters = extract_from_results(pred)

    mention_precision, mention_recall, mention_f1_score = evaluate_mention_scores(pred_mentions, gold_mentions)
    muc, b_cubed, ceafe, conll2012 = evaluate_coref_scores(pred_clusters, gold_clusters,
                                                                          pred_mention_to_clusters,
                                                                          gold_mention_to_clusters)
    with open(pred_output_name + "/report.txt", "w") as f:
        f.write("*** SCORES REPORT ***\n")
        f.write("EVALUATED FILE: " + gold_output_name+"\n")
        f.write("EVALUATED MODEL: \n*******************\n")
        f.write("\t|-number of samples: " + str(len(pred)) + "\n")
        f.write("\n\t|-mention scores: \n")
        f.write("\t|------precision: %.3f \n" % mention_precision)
        f.write("\t|------recall: %.3f \n" % mention_recall)
        f.write("\t|------f1 score: %.3f \n" % mention_f1_score)
        f.write("\n\t|-coreference scores: \n")
        f.write("\t|---muc(link-based metric evaluation):\n")
        f.write("\t|------precision: %.3f\n" % muc["precision"])
        f.write("\t|------recall: %.3f\n" % muc["recall"])
        f.write("\t|------f1 score: %.3f \n" % muc["f1_score"])
        f.write("\n\t|---b_cubed(mention-based metric evaluation):\n")
        f.write("\t|------precision: %.3f\n" % b_cubed["precision"])
        f.write("\t|------recall: %.3f\n" % b_cubed["recall"])
        f.write("\t|------f1 score: %.3f \n" % b_cubed["f1_score"])
        f.write("\n\t|---ceafe(entity-based metric evaluation):\n")
        f.write("\t|------precision: %.3f\n" % ceafe["precision"])
        f.write("\t|------recall: %.3f\n" % ceafe["recall"])
        f.write("\t|------f1 score: %.3f \n" % ceafe["f1_score"])
        f.write("\n\t|---CONLL-2012 (avg):\n")
        f.write("\t|------precision: %.3f\n" % conll2012["precision"])
        f.write("\t|------recall: %.3f\n" % conll2012["recall"])
        f.write("\t|------f1 scor: %.3f \n" % conll2012["f1_score"])

#creates output dir
def write_all_outputs(pred, gold, pred_output_name="output", gold_output_name="test.english.jsonlines"):
    try:
        subprocess.call("rm -r "+pred_output_name, shell=True)
    except:
        print("")
    subprocess.call("mkdir "+pred_output_name, shell=True)
    gold_output_path = pred_output_name + "/" + gold_output_name.strip(".jsonlines")
    pred_output_path = pred_output_name + "/" + pred_output_name

    write_docs_to_jsonlines(pred_output_path, pred)
    write_docs_to_jsonlines(gold_output_path, gold)

    jsonlines_to_html(gold_output_path)
    jsonlines_to_html(pred_output_path)

    jsonlines_to_conll(gold_output_path)
    jsonlines_to_conll(pred_output_path)

    write_scores_report(pred, gold, pred_output_name, gold_output_name)

#returns:
#mentions: [[(wstart, wend)]] list(docs) x list(mentions) x tuple(mention)
#clusters: [[((wstart, wend),()), (())...]] list(docs) x list(clusters) x tuple(mentions) x tuple(mention)
#mention_to_clusters: [{(wstart,wend):((wstart,wend),()...}] list(docs) x dict[tuple(mention)] : tuple(mentions) x tuple(mention)
def extract_from_results(docs):
    mentions = []
    clusters = []
    mention_to_clusters = []

    for doc in docs:
        mention_list = []
        cluster_list = []
        m2c = {}
        for cluster in doc["clusters"]:
            mention_list.extend([(m[0], m[1]) for m in cluster])
            cluster_tuple = tuple([(m[0], m[1]) for m in cluster])
            cluster_list.append(cluster_tuple)
            for mention in cluster:
                m2c[(mention[0], mention[1])] = cluster_tuple
        mentions.append(mention_list)
        clusters.append(cluster_list)
        mention_to_clusters.append(m2c)

    return mentions, clusters, mention_to_clusters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str, help="model name, from [lingmess, fastcoref], default:lingmess", default="lingmess")
    parser.add_argument("--custom_test_file", type=str, default="test.english.jsonlines", help="custom test set filename, default: test.english.jsonlines")
    parser.add_argument("--custom_output_path", type=str, default="output", help="custom output path filename, default: output")
    parser.add_argument("--cpu", action="store_true", help="to evaluate on cpu")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda:0" if not args.cpu else "cpu"
    if args.model == "lingmess":
        model = LingMessCoref(device=device)
    elif args.model == "fastcoref":
        model = FCoref(device=device)
    gold = extract_and_preprocess(input=args.custom_test_file)
    pred = predict(model, gold)
    write_all_outputs(pred, gold, pred_output_name=args.custom_output_path, gold_output_name=args.custom_test_file)

if __name__ == "__main__":
    main()