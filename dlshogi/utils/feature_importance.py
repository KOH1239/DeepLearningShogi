from cshogi import *
from cshogi.dlshogi import make_input_features, FEATURES1_NUM, FEATURES2_NUM

import numpy as np
import onnxruntime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, default='model', help='model file name')
parser.add_argument('sfen', type=str, help='position')
parser.add_argument('--svg', type=str)
args = parser.parse_args()

session = onnxruntime.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
FEATURES1_NUM += 2
board = Board(sfen=args.sfen)
features1 = np.zeros((41, FEATURES1_NUM, 9, 9), dtype=np.float32)
features2 = np.zeros((41, FEATURES2_NUM, 9, 9), dtype=np.float32)
make_input_features(board, features1, features2)


# 移動方向を表す定数
MOVE_DIRECTION = [
    UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT,
    UP2_LEFT, UP2_RIGHT,
    UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
] = range(20)


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


pos = []
i = 1
rank = 0
file = 0
j = 0

pieces_src = board.pieces
pieces_in_hand_src = board.pieces_in_hand

for sq in SQUARES:
	if pieces_src[sq] == NONE:
		continue

	file, rank = divmod(sq, 9)
	pieces_dst = pieces_src.copy()
	# if pieces_dst[sq] < 15:
	# 	pieces_dst[sq] = NONE
	# 	pieces_dst[board.king_square(BLACK)] = NONE
	# else:
	# 	pieces_dst[sq] = NONE
	# 	pieces_dst[board.king_square(WHITE)] = NONE
 
	pieces_dst[sq] = NONE

	board_dst = board.copy()
	board_dst.set_pieces(pieces_dst, pieces_in_hand_src)
	make_input_features(board_dst, features1[i], features2[i])
	pos.append((file, rank, pieces_src[sq]))
	i += 1

hand = []
for c in COLORS:
	for hp in HAND_PIECES:
		if pieces_in_hand_src[c][hp] == 0:
			continue

		pieces_in_hand_dst = (pieces_in_hand_src[0].copy(), pieces_in_hand_src[1].copy())
		pieces_in_hand_dst[c][hp] = 0
		
		# pieces_dst = pieces_src.copy()
		# pieces_dst[board.king_square(c)] = NONE

		board_dst = board.copy()
		board_dst.set_pieces(pieces_dst, pieces_in_hand_dst)
		make_input_features(board_dst, features1[i], features2[i])
		hand.append((c, hp, pieces_in_hand_src[c][hp]))
		i += 1

io_binding = session.io_binding()
io_binding.bind_cpu_input('input1', features1)
io_binding.bind_cpu_input('input2', features2)
io_binding.bind_output('output_policy')
io_binding.bind_output('output_value')
session.run_with_iobinding(io_binding)
y1, y2 = io_binding.copy_outputs_to_cpu()

importance = y2 - y2[0]


# 温度パラメータを適用した確率分布を取得
def softmax_temperature_with_normalize(logits, temperature):
    # 温度パラメータを適用
    logits /= temperature

    # 確率を計算(オーバーフローを防止するため最大値で引く)
    max_logit = max(logits)
    probabilities = np.exp(logits - max_logit)

    # 合計が1になるように正規化
    sum_probabilities = sum(probabilities)
    probabilities /= sum_probabilities

    return probabilities

color = board.turn
for policy in y1:
	# 合法手一覧
	legal_move_probabilities = np.empty(len(board.legal_moves), dtype=np.float32)
	for i in range(len(board.legal_moves)):
		move = list(board.legal_moves)[i]
		move_label = make_move_label(move, color)
		legal_move_probabilities[j] = policy[move_label]
	
	# Boltzmann分布
	probabilities = softmax_temperature_with_normalize(legal_move_probabilities, 1.0)
	
with open('importance.txt', 'w') as f:
	f.write("y2\n")
	for i in range(len(y2)):
		f.write(f'{y2[i]}\n')
	f.write(f"y1 probabilities: {probabilities}\n")
	f.write(f"probabilities_len: {len(probabilities)}\n")
	f.write(f"best_move_: {move_to_usi(list(board.legal_moves)[probabilities.argmax()])}\n")
	f.write("importance\n")
	for i in range(len(importance)):
		f.write(f'{importance[i]}\n')
	f.write(f"importance_sum: {importance.sum()}")

output = [['' for _ in range(9)] for _ in range(9)]
for i in range(len(pos)):
    file, rank, pc = pos[i]
    output[rank][8 - file] = format(float(importance[i + 1]), '.5f')
print('\n'.join(['\t'.join(row) for row in output]))

for i in range(len(hand)):
    c, hp, n = hand[i]
    symbol = HAND_PIECE_SYMBOLS[hp]
    if c == BLACK:
        symbol = symbol.upper()
    print(symbol, format(float(importance[len(pos) + 1 + i]), '.5f'), sep='\t')


def value_to_rgb(value):
	if value < 0 :
		r = 252 + int(value * 4)
		g = 252 + int(value * 147)
		b = 255 + int(value * 148)
	else:
		r = 252 - int(value * 162)
		g = 252 - int(value * 114)
		b = 255 - int(value * 57)
	return f"rgb({r},{g},{b})"

def to_svg(pos, hand, importance, scale=2.5):
	import xml.etree.ElementTree as ET

	width = 230
	height = 192

	svg = ET.Element("svg", {
		"xmlns": "http://www.w3.org/2000/svg",
		"version": "1.1",
		"xmlns:xlink": "http://www.w3.org/1999/xlink",
		"width": str(width * scale),
		"height": str(height * scale),
		"viewBox": "0 0 {} {}".format(width, height),
	})

	defs = ET.SubElement(svg, "defs")
	for piece_def in SVG_PIECE_DEFS:
		defs.append(ET.fromstring(piece_def))

	for i in range(len(pos)):
		file, rank, pc = pos[i]
		value = float(importance[i + 1])
		ET.SubElement(svg, "rect", {
			"x": str(20.5 + (8 - file) * 20),
			"y": str(10.5 + rank * 20),
			"width": str(20),
			"height": str(20),
			"fill": value_to_rgb(value)
		})

	svg.append(ET.fromstring(SVG_SQUARES))
	svg.append(ET.fromstring(SVG_COORDINATES))

	for i in range(len(pos)):
		file, rank, pc = pos[i]
		x = 20.5 + (8 - file) * 20
		y = 10.5 + rank * 20
		value = float(importance[i + 1])

		ET.SubElement(svg, "use", {
			"xlink:href": "#{}".format(SVG_PIECE_DEF_IDS[pc]),
			"x": str(x),
			"y": str(y),
		})
		e = ET.SubElement(svg, "text", {
			"font-family": "serif",
			"font-size": "5",
			"stroke-width": "1",
			"stroke": "#fff",
			"fill": "#000",
			"paint-order": "stroke",
			"x": str(x + 12),
			"y": str(y + 19)
		})
		e.text = format(abs(value), '.2f')[1:]

	hand_by_color = [[], []]
	for i in range(len(hand)):
		c, hp, n = hand[i]
		hand_by_color[c].append((i, hp, n))

	hand_pieces = [[], []]
	for c in COLORS:
		i = 0
		for index, hp, n in hand_by_color[c]:
			if n >= 11:
				hand_pieces[c].append((i, NUMBER_JAPANESE_KANJI_SYMBOLS[n % 10], None))
				i += 1
				hand_pieces[c].append((i, NUMBER_JAPANESE_KANJI_SYMBOLS[10], None))
				i += 1
			elif n >= 2:
				hand_pieces[c].append((i, NUMBER_JAPANESE_KANJI_SYMBOLS[n], None))
				i += 1
			if n >= 1:
				hand_pieces[c].append((i, HAND_PIECE_JAPANESE_SYMBOLS[hp], index))
				i += 1
		i += 1
		hand_pieces[c].append((i, "手", None))
		i += 1
		hand_pieces[c].append((i, "先" if c == BLACK else "後", None))
		i += 1
		hand_pieces[c].append(( i, "☗" if c == BLACK else "☖", None))

	for c in COLORS:
		if c == BLACK:
			x = 214
			y = 190
			x_rect = 214
			y_rect = 178
		else:
			x = -16
			y = -10
			x_rect = 2
			y_rect = 8
		scale = 1
		if len(hand_pieces[c]) + 1 > 13:
			scale = 13.0 / (len(hand_pieces[c]) + 1)
		for i, text, index in hand_pieces[c]:
			if index is not None:
				value = float(importance[len(pos) + 1 + index])
				ET.SubElement(svg, "rect", {
					"x": str(x_rect),
					"y": str(y_rect + 14 * scale * i * (-1 if c == BLACK else 1)),
					"width": str(14),
					"height": str(14 * scale),
					"fill": value_to_rgb(value)
				})

			e = ET.SubElement(svg, "text", {
				"font-family": "serif",
				"font-size": str(14 * scale),
			})
			e.set("x", str(x))
			e.set("y", str(y - 14 * scale * i))
			if c == WHITE:
				e.set("transform", "rotate(180)")
			e.text = text

			if index is not None:
				e = ET.SubElement(svg, "text", {
					"font-family": "serif",
					"font-size": "5",
					"stroke-width": "1",
					"stroke": "#fff",
					"fill": "#000",
					"paint-order": "stroke",
					"x": str(x_rect + 7),
					"y": str(y_rect + 14 * scale * i * (-1 if c == BLACK else 1) + 13.5 * scale)
				})
				e.text = format(abs(value), '.2f')[1:]

	return ET.ElementTree(svg)


if args.svg:
	with open(args.svg, 'wb') as f:
		svg = to_svg(pos, hand, importance)
		svg.write(f)
