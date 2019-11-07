#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from six.moves import xrange
import paddle.fluid as fluid
import AUC
from model.ernie import ErnieModel
import os

def create_model(args, pyreader_name, ernie_config, is_prediction=False):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1, 1],
                [-1, 1]],
        dtypes=['int64', 'int64', 'int64', 'float32', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, input_mask, labels,
     qids) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    cls_feats = ernie.get_pooled_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        size=args.num_labels,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    if is_prediction:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, input_mask.name
        ]
        return pyreader, probs, feed_targets_name
    
    def focal_loss(logits, labels):
        probs = fluid.layers.softmax(logits)
        probs_0 = fluid.layers.reshape(fluid.layers.slice(
             input=probs, axes=[1], starts=[0], ends=[1]) ,shape=[-1,1])
        probs_1 = fluid.layers.reshape(fluid.layers.slice(
            input=probs, axes=[1], starts=[1], ends=[2]) ,shape=[-1,1])
        gamma = 2.0
        f_labels = fluid.layers.cast(labels, 'float32')
        ce_loss = -10 * f_labels * fluid.layers.log(probs_1) - (1 - f_labels) * fluid.layers.log(probs_0)
        # ce_loss = 0 - 1 * f_labels * fluid.layers.pow(probs_0, gamma) * fluid.layers.log(probs_1) \
        #    - (1 - f_labels) * fluid.layers.pow(probs_1, gamma) * fluid.layers.log(probs_0) 
        return ce_loss, probs
    
    ce_loss, probs = focal_loss(logits=logits, labels=labels)

    # ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
    #    logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    graph_vars = {
        "loss": loss,
        "probs": probs,
        "accuracy": accuracy,
        "labels": labels,
        "num_seqs": num_seqs,
        "qids": qids
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def evaluate_mrr(preds):
    last_qid = None
    total_mrr = 0.0
    qnum = 0.0
    rank = 0.0
    correct = False
    for qid, score, label in preds:
        if qid != last_qid:
            rank = 0.0
            qnum += 1
            correct = False
            last_qid = qid

        rank += 1
        if not correct and label != 0:
            total_mrr += 1.0 / rank
            correct = True

    return total_mrr / qnum


def evaluate_map(preds):
    def singe_map(st, en):
        total_p = 0.0
        correct_num = 0.0
        for index in xrange(st, en):
            if int(preds[index][2]) != 0:
                correct_num += 1
                total_p += correct_num / (index - st + 1)
        if int(correct_num) == 0:
            return 0.0
        return total_p / correct_num

    last_qid = None
    total_map = 0.0
    qnum = 0.0
    st = 0
    for i in xrange(len(preds)):
        qid = preds[i][0]
        if qid != last_qid:
            qnum += 1
            if last_qid != None:
                total_map += singe_map(st, i)
            st = i
            last_qid = qid

    total_map += singe_map(st, len(preds))
    return total_map / qnum


def evaluate(exe, test_program, test_pyreader, graph_vars, eval_phase):
    train_fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["num_seqs"].name
    ]

    if eval_phase == "train":
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list)
        ret = {"loss": np.mean(outputs[0]), "accuracy": np.mean(outputs[1])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[3][0])
        return ret

    test_pyreader.start()
    total_cost, total_acc, total_num_seqs, total_label_pos_num, total_pred_pos_num, total_correct_num, \
            total_label_neg_num, total_pred_neg_num, total_neg_correct_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    qids, labels, scores = [], [], []
    time_begin = time.time()

    fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["probs"].name, graph_vars["labels"].name,
        graph_vars["num_seqs"].name, graph_vars["qids"].name
    ]
    batch_id = 0
    if eval_phase == "infer":
        try:
            os.remove("predict_scores.txt")
        except:
            pass

    while True:
        try:
            batch_id += 1
            np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                program=test_program, fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_acc += np.sum(np_acc * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            #####qids.extend(np_qids.reshape(-1).tolist())
            scores.extend(np_probs[:, 1].reshape(-1).tolist())
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            total_label_pos_num += np.sum(np_labels)
            total_pred_pos_num += np.sum(np_preds)
            total_correct_num += np.sum(np.dot(np_preds, np_labels))
            
            np_labels = np_labels.reshape(-1)
            total_neg_correct_num += np.sum((np_preds + np_labels)==0)
            total_label_neg_num += np.sum(np_labels == 0)
            total_pred_neg_num += np.sum(np_preds == 0)

            if eval_phase == "infer": 
                if (batch_id % 100) == 0:
                    fs = open("predict_scores.txt", "a+")
                    for score in scores:
                        fs.write(str(score) + "\n")
                    fs.close()
                    qids, labels, scores = [], [], []
                    print("processed batch num: ", batch_id)

        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    
    if eval_phase == "infer":
        fs = open("predict_scores.txt", "a+")
        for score in scores:
            fs.write(str(score) + "\n")
        fs.close()
        return

    time_end = time.time()
    
    pos_r = total_correct_num / total_label_pos_num
    pos_p = total_correct_num / total_pred_pos_num
    pos_f = 2 * pos_p * pos_r / (pos_p + pos_r)

    neg_r = total_neg_correct_num / total_label_neg_num
    neg_p = total_neg_correct_num / total_pred_neg_num
    neg_f = 2 * neg_p * neg_r / (neg_p + neg_r)

    auc = AUC.auc(labels, scores)

    if len(qids) == 0:
        print(
                "[%s evaluation] ave loss: %f, ave acc: %f, pos p: %f, pos r: %f, pos f1: %f, neg p: %f, neg r: %f, neg f1: %f, auc: %f, data_num: %d, elapsed time: %f s"
            % (eval_phase, total_cost / total_num_seqs, total_acc /
               total_num_seqs, pos_p, pos_r, pos_f, neg_p, neg_r, neg_f, auc, total_num_seqs, time_end - time_begin))
    else:

        assert len(qids) == len(labels) == len(scores)
        preds = sorted(
            zip(qids, scores, labels), key=lambda elem: (elem[0], -elem[1]))
        mrr = evaluate_mrr(preds)
        map = evaluate_map(preds)

        print(
                "[%s evaluation] ave loss: %f, ave_acc: %f, mrr: %f, map: %f, p: %f, r: %f, f1: %f, auc: %f, data_num: %d, elapsed time: %f s"
            % (eval_phase, total_cost / total_num_seqs,
               total_acc / total_num_seqs, mrr, map, p, r, f, auc, total_num_seqs,
               time_end - time_begin))
