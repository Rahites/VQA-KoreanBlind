import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word


def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0


def path2rest(path, split, annotations, label2ans):
    iid = int(path.split("/")[-1].split("_")[-1][:-4])

    with open(path, "rb") as fp:
        binary = fp.read()

    _annot = annotations[split][iid]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas] if "test" not in split else list(list())
    answer_labels = (
        [a["labels"] for a in answers] if "test" not in split else list(list())
    )
    answer_scores = (
        [a["scores"] for a in answers] if "test" not in split else list(list())
    )
    answers = (
        [[label2ans[l] for l in al] for al in answer_labels]
        if "test" not in split
        else list(list())
    )

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, split]


def make_arrow(root, dataset_root):
    with open(f"{root}/OpenEnded_aihub_train_questions.json", "r") as fp:
        questions_train = json.load(fp)["questions"]
    with open(f"{root}/OpenEnded_aihub_valid_questions.json", "r") as fp:
        questions_val = json.load(fp)["questions"]
    with open(f"{root}/OpenEnded_aihub_test_questions.json", "r") as fp:
        questions_test = json.load(fp)["questions"]
    # with open(f"{root}/v2_OpenEnded_mscoco_test-dev2015_questions.json", "r") as fp:
        # questions_test_dev = json.load(fp)["questions"]

    with open(f"{root}/aihub_train_annotation.json", "r") as fp:
        annotations_train = json.load(fp)["annotations"]
    with open(f"{root}/aihub_valid_annotation.json", "r") as fp:
        annotations_val = json.load(fp)["annotations"]

    annotations = dict()

    for split, questions in zip(
        ["train", "val", "test"], # delete "test-dev"
        [
            questions_train,
            questions_val,
            questions_test
        ],
    ):
        _annot = defaultdict(dict)
        for q in tqdm(questions):
            _annot[q["image_id"]][q["question_id"]] = [q["question"]]

        annotations[split] = _annot

    all_major_answers = list()

    for split, annots in zip(
        ["train", "val"], [annotations_train, annotations_val],
    ):
        # _annot = annotations[split]
        for q in tqdm(annots):
            all_major_answers.append(q["multiple_choice_answer"])

    # ai hub 데이터가 깔끔해서 정규화가 필요 없다고 생각? 
    # all_major_answers = [normalize_word_kor(word) for word in tqdm(all_major_answers)]
    all_major_answers = [word for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 8} # here
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())

    print('counter : ', len(counter))
    # print('ans2label : ', len(ans2label))
    # print('label2ans')

    for split, annots in zip(
        ["train", "val"], [annotations_train, annotations_val],
    ):
        _annot = annotations[split]

        for q in tqdm(annots):
            answers = q["answers"]
            answer_count = {}
            for answer in answers:
                answer_ = answer["answer"]
                answer_count[answer_] = answer_count.get(answer_, 0) + 1

            labels = []
            scores = []
            
            for answer in answer_count:
                if answer not in ans2label:
                    continue
                labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)

            _annot[q["image_id"]][q["question_id"]].append(
                {"labels": labels, "scores": scores,}
            )
        
        # print('error : ', error)
        # print(_annot)
        # print('all_major_answers 개수 : ', len(all_major_answers)) # 125290
        # print('_annot 개수 : ', len(_annot)) # 3750
        # print('error 개수 : ', len(error)) # 2421
        # print('annotations 개수 : ', len(annotations)) # 3
    # print(len(annotations['train']))

    for split in ["train", "val"]:
        filtered_annot = dict()
        for ik, iv in annotations[split].items(): # ik : image_id, iv : {question_id : [question, {label, scores}]}
            new_q = dict()
            for qk, qv in iv.items(): # qk : question_id, qv : ['장갑을 낀 사람은 모자를 쓰고 있습니까?', {'labels': [5], 'scores': [1.0]}]
                if len(qv[1]["labels"]) != 0: # 라벨이 없는 애들 제거 
                    new_q[qk] = qv
            if len(new_q) != 0:
                filtered_annot[ik] = new_q
        annotations[split] = filtered_annot

    # print(len(annotations['train']))
    # print(annotations['val'])

    # import sys
    # sys.exit()

    for split in [
        "train",
        "val",
        "test"
    ]:
        annot = annotations[split]
        split_name = {
            "train": "aihub_train",
            "val": "aihub_valid",
            "test": "aihub_test"
        }[split]
        paths = list(glob(f"{root}/{split_name}/*.jpg"))
        random.shuffle(paths)
        # print('annot : ', annot)
        # print('path : ', paths[0])
        annot_paths = [
            path
            for path in paths
            if int(path.split("/")[-1].split("_")[-1][:-4]) in annot # check
        ]
        if len(paths) == len(annot_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
            for path_tmp in paths:
                if path_tmp not in annot_paths:
                    print(path_tmp)

        print(
            len(paths), len(annot_paths), len(annot),
        )


        bs = [
            path2rest(path, split, annotations, label2ans) for path in tqdm(annot_paths)
        ]

        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/vqav2_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

    table = pa.ipc.RecordBatchFileReader(
        pa.memory_map(f"{dataset_root}/vqav2_val.arrow", "r")
    ).read_all()

    pdtable = table.to_pandas()

    df1 = pdtable[:-1000]
    df2 = pdtable[-1000:]

    df1 = pa.Table.from_pandas(df1)
    df2 = pa.Table.from_pandas(df2)

    with pa.OSFile(f"{dataset_root}/vqav2_trainable_val.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, df1.schema) as writer:
            writer.write_table(df1)

    with pa.OSFile(f"{dataset_root}/vqav2_rest_val.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, df2.schema) as writer:
            writer.write_table(df2)
