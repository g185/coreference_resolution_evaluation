from transformers import LongformerModel, AutoTokenizer, RobertaModel
import torch
import json
cosine = torch.nn.CosineSimilarity(1)

def avg_cosine_similarity(cluster, element):
    element_tensor = element[0]
    element_words = element[1]
    if cluster == []:
        return
    else:
        print("\nelement,", element_words)
        for tensors, words in cluster:
            if tensors.shape[0] == 1:
                print("\tcomparing with single span element" + str(words) + " with cosine similarity " + str([cosine(element_tensor, tensors).detach()]))
            else:
                print("\tcomparing with multibpe span element" + str(words))
                for idx, elem_tensor in enumerate(tensors):
                    print("\t\tcomparing with subspan " + str(words[idx:idx + 1]) + " with cosine similarity " + str(cosine(element_tensor, elem_tensor).detach()))




tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096", use_fast= True, add_prefix_space=True, return_tensors='pt')

doc = {"doc_key": "test_obama" , "text":[["Barack", "Hussein", "Obama", "II", "is", "an", "American", "politician", "who", "served", "as", "the", "44th", "president", "of", "the", "United", "States", "from", "2009", "to", "2017", ".", "A", "member", "of", "the", "Democratic", "Party", ",", "Obama", "was", "the", "first", "African-American", "president", "of", "the", "United", "States", ".", "He", "previously", "served", "as", "a", "U", ".S", ".", "senator", "from", "Illinois", "from", "2005", "to", "2008", "and", "as", "an", "Illinois", "state", "senator", "from", "1997", "to", "2004", ",", "and", "previously", "worked", "as", "a", "civil", "rights", "lawyer", "before", "entering", "politics", ".", "Obama", "was", "born", "in", "Honolulu", ",", "Hawaii", ".", "After", "graduating", "from", "Columbia", "University", "in", "1983", ",", "he", "worked", "as", "a", "community", "organizer", "in", "Chicago", ".", "In", "1988", ",", "he", "enrolled", "in", "Harvard", "Law", "School", ",", "where", "he", "was", "the", "first", "black", "president", "of", "the", "Harvard", "Law", "Review", ".", "After", "graduating", ",", "he", "became", "a", "civil", "rights", "attorney", "and", "an", "academic", ",", "teaching", "constitutional", "law", "at", "the", "University", "of", "Chicago", "Law", "School", "from", "1992", "to", "2004", ".", "Turning", "to", "elective", "politics", ",", "he", "represented", "the", "13th", "district", "in", "the", "Illinois", "Senate", "from", "1997", "until", "2004", ",", "when", "he", "ran", "for", "the", "U", ".S", ".", "Senate", ".", "Obama", "received", "national", "attention", "in", "2004", "with", "his", "March", "Senate", "primary", "win", ",", "his", "well-received", "July", "Democratic", "National", "Convention", "keynote", "address", ",", "and", "his", "landslide", "November", "election", "to", "the", "Senate", ".", "In", "2008", ",", "after", "a", "close", "primary", "campaign", "against", "Hillary", "Clinton", ",", "he", "was", "nominated", "by", "the", "Democratic", "Party", "for", "president", "and", "chose", "Joe", "Biden", "as", "his", "running", "mate", ".", "Obama", "was", "elected", "over", "Republican", "nominee", "John", "McCain", "in", "the", "presidential", "election", "and", "was", "inaugurated", "on", "January", "20", ",", "2009", ".", "Nine", "months", "later", ",", "he", "was", "named", "the", "2009", "Nobel", "Peace", "Prize", "laureate", ",", "a", "decision", "that", "drew", "a", "mixture", "of", "praise", "and", "criticism", ".", "Obama's", "first-term", "actions", "addressed", "the", "global", "financial", "crisis", "and", "included", "a", "major", "stimulus", "package", ",", "a", "partial", "extension", "of", "George", "W", ".", "Bush's", "tax", "cuts", ",", "legislation", "to", "reform", "health", "care", ",", "a", "major", "financial", "regulation", "reform", "bill", ",", "and", "the", "end", "of", "a", "major", "US", "military", "presence", "in", "Iraq", ".", "Obama", "also", "appointed", "Supreme", "Court", "justices", "Sonia", "Sotomayor", "and", "Elena", "Kagan", ",", "the", "former", "being", "the", "first", "Hispanic", "American", "on", "the", "Supreme", "Court", ".", "He", "ordered", "the", "counterterrorism", "raid", "which", "killed", "Osama", "bin", "Laden", "and", "downplayed", "Bush's", "counterinsurgency", "model", ",", "expanding", "air", "strikes", "and", "making", "extensive", "use", "of", "special", "forces", "while", "encouraging", "greater", "reliance", "on", "host-government", "militaries", ".", "After", "winning", "re-election", "by", "defeating", "Republican", "opponent", "Mitt", "Romney", ",", "Obama", "was", "sworn", "in", "for", "a", "second", "term", "on", "January", "20", ",", "2013", ".", "In", "his", "second", "term", ",", "Obama", "took", "steps", "to", "combat", "climate", "change", ",", "signing", "a", "major", "international", "climate", "agreement", "and", "an", "executive", "order", "to", "limit", "carbon", "emissions", ".", "Obama", "also", "presided", "over", "the", "implementation", "of", "the", "Affordable", "Care", "Act", "and", "other", "legislation", "passed", "in", "his", "first", "term", ",", "and", "he", "negotiated", "a", "nuclear", "agreement", "with", "Iran", "and", "normalized", "relations", "with", "Cuba", ".", "The", "number", "of", "American", "soldiers", "in", "Afghanistan", "fell", "dramatically", "during", "Obama's", "second", "term", ",", "though", "U", ".S", ".", "soldiers", "remained", "in", "Afghanistan", "throughout", "Obama's", "presidency", ".", "During", "Obama's", "terms", "as", "president", ",", "the", "United", "States'", "reputation", "abroad", "and", "the", "American", "economy", "improved", "significantly", ",", "although", "the", "country", "experienced", "high", "levels", "of", "partisan", "divide", ".", "As", "the", "first", "person", "of", "color", "elected", "president", "Obama", "faced", "racist", "sentiments", "and", "was", "the", "target", "of", "numerous", "conspiracy", "theories", ".", "Obama", "left", "office", "on", "January", "20", ",", "2017", ",", "and", "continues", "to", "reside", "in", "Washington", ",", "D", ".C", "."]], "clusters": [[[30, 30], [79, 79], [184, 184], [245, 245], [342, 342], [410, 410], [429, 429], [452, 452], [561, 561], [547, 548], [0, 3], [12, 17], [291, 291], [496, 496], [509, 509], [513, 513]], [[13, 17], [35, 39], [120, 120], [235, 235], [516, 516], [547, 547]], [[27, 28], [232, 233]], [[51, 51], [59, 59], [167, 167]], [[83, 83]]], "pronouns": [[41, 41], [95, 95], [107, 107], [115, 115], [130, 130], [160, 160], [175, 175], [191, 191], [197, 197], [207, 207], [227, 227], [241, 241], [270, 270], [366, 366], [425, 425], [468, 468], [473, 473]], "EOS":[[22, 22], [40, 40], [48, 48], [78, 78], [86, 86], [103, 103], [126, 126], [154, 154], [181, 181], [183, 183], [214, 214], [244, 244], [265, 265], [290, 290], [312, 312], [341, 341], [365, 365], [399, 399], [423, 423], [451, 451], [485, 485], [503, 503], [511, 511], [539, 539], [560, 560], [579, 579]]}

tokenized = tokenizer(doc["text"][0], is_split_into_words=True, return_tensors='pt', return_offsets_mapping=True)

clusters_in_tokens = doc["clusters"]

clusters_in_ids = [[(tokenized.word_to_tokens(start).start,
                                 tokenized.word_to_tokens(end).end - 1)
                                for start, end in cluster] for cluster in doc["clusters"]]

clusters_in_words = [[doc["text"][0][start:end+4]
                                for start, end in cluster] for cluster in doc["clusters"]]

model = LongformerModel.from_pretrained("allenai/longformer-large-4096")
model3 = RobertaModel.from_pretrained("roberta-large")
tokenizer3 = AutoTokenizer.from_pretrained("roberta-large")
hs = model(tokenized["input_ids"], output_hidden_states=True)
lhs = hs["last_hidden_state"][0]
lhs2 = hs["hidden_states"][-2][0]
lhs3 = hs["hidden_states"][-3][0]
lhs4 = hs["hidden_states"][-4][0]

coreferences = []
coreference_hidden_states = []
for cluster_of_words, cluster_of_ids in zip(clusters_in_words, clusters_in_ids ):
    cluster_hidden_states = []
    cluster = {}
    for span_of_words, span_of_ids in zip(cluster_of_words, cluster_of_ids):

        if span_of_ids[0] == span_of_ids[1]:
            cluster[span_of_ids] = (lhs[span_of_ids[0]].unsqueeze(0), span_of_words)
            cluster_hidden_states.append((lhs[span_of_ids[0]].unsqueeze(0), span_of_words))
        else:
            cluster[span_of_ids] = (lhs[span_of_ids[0]:span_of_ids[1] + 1], span_of_words)
            cluster_hidden_states.append((lhs[span_of_ids[0]:span_of_ids[1] + 1], span_of_words))

    coreferences.append(cluster)
    coreference_hidden_states.append(cluster_hidden_states)

###
pronouns_in_ids = [(tokenized.word_to_tokens(start).start,
                                 tokenized.word_to_tokens(end).end - 1)
                                for start, end in doc["pronouns"]]

pronouns = [doc["text"][0][start:end + 4] for start,end in doc["pronouns"]]

points_in_ids = [tokenized.word_to_tokens(start).start
                                for start, end in doc["EOS"]]

def extract_antecedent_coreferences(coreferences, idx):
    coref_temp = []
    min = sorted([elem for elem in points_in_ids if elem < idx[0]])
    try:
        start_of_previous_sentence = min[-3]
    except:
        start_of_previous_sentence = 0

    for cluster in coreferences:
        t = {}
        for key, val in cluster.items():
            if key[0] < idx[0]:
                t[key] = val
        coref_temp.append(t)
    return coref_temp


for pronoun_idx, pronoun, idx in zip(pronouns_in_ids, pronouns, doc["pronouns"]):
    antecedent_coreferences = extract_antecedent_coreferences(coreferences, pronoun_idx)
    lhs_of_pronoun = lhs[pronoun_idx[0]]
    max_cosine_cluster = None
    max_cos_value = 0
    i = 0
    for cluster in antecedent_coreferences:
        sim = 0
        for span_idxs, (h, w) in cluster.items():
            if span_idxs[0] == span_idxs[1]:
                sim += torch.mean(cosine(lhs[pronoun_idx[0]], lhs[span_idxs[0]].unsqueeze(0)))
                sim += torch.mean(cosine(lhs2[pronoun_idx[0]], lhs2[span_idxs[0]].unsqueeze(0)))
                sim += torch.mean(cosine(lhs3[pronoun_idx[0]], lhs3[span_idxs[0]].unsqueeze(0)))
                sim += torch.mean(cosine(lhs4[pronoun_idx[0]], lhs4[span_idxs[0]].unsqueeze(0)))
            else:
                sim += torch.mean(cosine(lhs[pronoun_idx[0]], lhs[span_idxs[0]:span_idxs[1]+1]))
                sim += torch.mean(cosine(lhs2[pronoun_idx[0]], lhs2[span_idxs[0]:span_idxs[1]+1]))
                sim += torch.mean(cosine(lhs3[pronoun_idx[0]], lhs3[span_idxs[0]:span_idxs[1]+1]))
                sim += torch.mean(cosine(lhs4[pronoun_idx[0]], lhs4[span_idxs[0]:span_idxs[1]+1]))
        if len(cluster.items()) > 0 :
            sim = sim / (len(cluster.items())*4)
            print(pronoun, "has similarity", sim.item(), [value[1][0] for key, value in cluster.items()])
            if sim > max_cos_value:
                max_cos_value = sim
                max_cosine_cluster = ([value[1] for key, value in cluster.items()], sim.item(), 0, i)
        i += 1
    print(pronoun, " is in cluster",  max_cosine_cluster)
    doc["clusters"][max_cosine_cluster[3]].extend([idx])
doc["sentences"] = doc.pop("text")
with open("/mnt/ssd/coreference_resolution_evaluation/test.jsonlines", "a") as f:
    f.write("\n")
    json.dump(doc, f)
a = 1


