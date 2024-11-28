import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from dlshogi.common import *
from dlshogi import serializers
from dlshogi.data_loader import DataLoader
from dlshogi.network.policy_value_network import policy_value_network
from cshogi import *
import cshogi
from dlshogi import cppshogi

import argparse
import random
import os
import sys

import logging

# 移動を表すラベルの数
MOVE_PLANES_NUM = len(MOVE_DIRECTION) + len(HAND_PIECES)
MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81

# 移動を表すラベルを作成
def make_move_label(move, color):
    if not move_is_drop(move):  # 駒の移動
        to_sq = move_to(move)
        from_sq = move_from(move)

        # 後手の場合盤を回転
        if color == WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        # 移動方向
        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y < 0:
            if dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_RIGHT
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_LEFT
            elif dir_x < 0:
                move_direction = UP_RIGHT
            else:  # dir_x > 0
                move_direction = UP_LEFT
        elif dir_y == 0:
            if dir_x < 0:
                move_direction = RIGHT
            else:  # dir_x > 0
                move_direction = LEFT
        else:  # dir_y > 0
            if dir_x == 0:
                move_direction = DOWN
            elif dir_x < 0:
                move_direction = DOWN_RIGHT
            else:  # dir_x > 0
                move_direction = DOWN_LEFT

        # 成り
        if move_is_promotion(move):
            move_direction += 10
    else:  # 駒打ち
        to_sq = move_to(move)
        # 後手の場合盤を回転
        if color == WHITE:
            to_sq = 80 - to_sq

        # 駒打ちの移動方向
        move_direction = len(MOVE_DIRECTION) + move_drop_hand_piece(move)

    return move_direction * 81 + to_sq



def main(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='model', help='model file name')
    parser.add_argument('test_data', type=str, help='test data file')
    parser.add_argument('--testbatchsize', type=int, default=1024, help='Number of positions in each test mini-batch')
    parser.add_argument('--network', default='resnet10_swish', help='network type')
    parser.add_argument('--log', default=None, help='log file path')
    parser.add_argument('--val_lambda', type=float, default=0.333, help='regularization factor')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--onnx', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print('Load model from', args.model)
    if args.onnx:
        import onnxruntime
        session = onnxruntime.InferenceSession(args.model)
    else:
        model = policy_value_network(args.network)
        model.to(device)
        serializers.load_npz(args.model, model)
        model.eval()

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

    logging.debug('read test data')
    logging.debug(args.test_data)
    test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

    logging.info('test position num = {}'.format(len(test_data)))

    test_dataloader = DataLoader(test_data, args.testbatchsize, torch.device("cpu") if args.onnx else device)

    def accuracy(y, t):
        return (torch.max(y, 1)[1] == t).sum().item() / len(t)

    def binary_accuracy(y, t):
        pred = y >= 0
        truth = t >= 0.5
        return pred.eq(truth).sum().item() / len(t)

    itr_test = 0
    sum_test_loss1 = 0
    sum_test_loss2 = 0
    sum_test_loss3 = 0
    sum_test_loss = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    sum_test_entropy1 = 0
    sum_test_entropy2 = 0
    with torch.no_grad():
        for x1, x2, t1, t2, value in test_dataloader:
            if args.onnx:
                io_binding = session.io_binding()
                io_binding.bind_cpu_input('input1', x1.numpy())
                io_binding.bind_cpu_input('input2', x2.numpy())
                io_binding.bind_output('output_policy')
                io_binding.bind_output('output_value')
                # io_binding.bind_output('att_p')
                # io_binding.bind_output('att_v')
                session.run_with_iobinding(io_binding)
                y1, y2, _, _ = io_binding.copy_outputs_to_cpu()
                y1 = torch.from_numpy(y1).to(device)
                y2 = torch.from_numpy(y2).to(device)
                # a1 = torch.from_numpy(a1).to(device)
                # a2 = torch.from_numpy(a2).to(device)
                y2 = torch.log(y2 / (1 - y2))
                t1 = t1.to(device)
                t2 = t2.to(device)
                value = value.to(device)
            else:
                y1, y2, a1, a2 = model(x1, x2)
            
            
            
            itr_test += 1
            loss1 = cross_entropy_loss(y1, t1).mean()
            loss2 = bce_with_logits_loss(y2, t2)
            loss3 = bce_with_logits_loss(y2, value)
            loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3
            sum_test_loss1 += loss1.item()
            sum_test_loss2 += loss2.item()
            sum_test_loss3 += loss3.item()
            sum_test_loss += loss.item()
            sum_test_accuracy1 += accuracy(y1, t1)
            sum_test_accuracy2 += binary_accuracy(y2, t2)

            entropy1 = (- F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1)
            sum_test_entropy1 += entropy1.mean().item()

            p2 = y2.sigmoid()
            #entropy2 = -(p2 * F.log(p2) + (1 - p2) * F.log(1 - p2))
            log1p_ey2 = F.softplus(y2)
            entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
            sum_test_entropy2 +=entropy2.mean().item()
            
            logits = y1.cpu().numpy()
            value = torch.sigmoid(y2).cpu().numpy()
            att_p = model.att_p.cpu().numpy()
            att_v = model.att_v.cpu().numpy()

            board = cshogi.Board()
            board.set_hcp(test_data[itr_test]['hcp'])
            print(board.sfen())

            logits_ = torch.zeros(len(board.legal_moves), dtype=torch.float32)
            for j, m in enumerate(board.legal_moves):
                label = make_move_label(m, board.turn)
                logits_[j] = logits[0, label].item()
            probabilities = F.softmax(logits_, dim=0)

            print('policy')
            for m, p in zip(board.legal_moves, probabilities):
                print(f"{cshogi.move_to_usi(m)}\t{p}")

            print(f"value\t{value[0]}")
            print('policy attention map')
            for rank in range(9):
                for file in reversed(range(9)):
                    print(att_p[0, 0, file, rank], end='\t')
                print()

            print('value attention map')
            for rank in range(9):
                for file in reversed(range(9)):
                    print(att_v[0, 0, file, rank], end='\t')
                print()
            

        logging.info('test_loss = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test accuracy = {:.08f}, {:.08f}, test entropy = {:.08f}, {:.08f}'.format(
            sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss3 / itr_test, sum_test_loss / itr_test,
            sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test,
            sum_test_entropy1 / itr_test, sum_test_entropy2 / itr_test))
        
        
        

if __name__ == '__main__':
    main(*sys.argv[1:])