# -*- coding: UTF-8 -*-
import sys

if __name__ == "__main__":
    split_type = sys.argv[1]
    sentence_type = sys.argv[2]
    data_path = sys.argv[3]

    dataset_fixes = {"\x83\xc2": "", "-LRB-":"(", "-RRB-":")", "\xc3\x82\xc2\xa0":" "}

    #read dataset split
    dataset_split = {}
    with open(data_path + "datasetSplit.txt", "r") as f:
        next(f)
        for line in f:
            line_parts = line.strip().split(",")
            dataset_split[line_parts[0].strip()] = line_parts[1].strip()

    # read relevant sentences
    sentences = []
    with open(data_path + "datasetSentences.txt", "r") as f:
        next(f)
        for line in f:
            line_parts = line.strip().split("\t")
            if len(line_parts) != 2:
                raise ValueError("Unexpected file format")
            if dataset_split[line_parts[0]] == split_type:
                sentence = line_parts[1]
                for fix in dataset_fixes:
                    sentence = sentence.replace(fix, dataset_fixes[fix])
                sentences.append(sentence)


    # read sentiment labels
    sentiment_labels = {}
    with open(data_path + "sentiment_labels.txt", "r") as f:
        next(f)
        for line in f:
            line_parts = line.strip().split("|")
            if len(line_parts) != 2:
                raise ValueError("Unexpected file format")
            sentiment_labels[line_parts[0]] = float(line_parts[1])

    # read the phrases
    phrases = {}
    with open(data_path + "dictionary.txt", "r") as f:
        for line in f:
            line_parts = line.strip().split("|")
            if len(line_parts) != 2:
                raise ValueError("Unexpected file format")
            phrases[line_parts[0]] = sentiment_labels[line_parts[1]]

    # print the labels and sentences/phrases
    if sentence_type == "full":
        for sentence in sentences:
            print str(phrases[sentence]) + "\t" + sentence
    elif sentence_type == "all":
        for phrase in phrases:
            print_phrase = False
            for sentence in sentences:
                if sentence.find(phrase) >= 0:
                    print_phrase = True
                    break
            if print_phrase:
                print str(phrases[phrase]) + "\t" + phrase
