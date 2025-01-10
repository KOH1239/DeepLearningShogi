import argparse
import os
import glob
from itertools import zip_longest

from cshogi import *
from cshogi import KIF
from cshogi import CSA
from dlshogi.utils.remove_comment import remove_no_need_comment

parser = argparse.ArgumentParser()
parser.add_argument("kif_dir")
parser.add_argument("csa_dir")
parser.add_argument("--encoding")
args = parser.parse_args()

for path in glob.glob(os.path.join(args.kif_dir, "**", "*.kif"), recursive=True):
    try:
        kif = KIF.Parser.parse_file(path)
        relpath = os.path.relpath(path, args.kif_dir)
        csa_path = os.path.join(args.csa_dir, relpath)
        dirname, filename = os.path.split(csa_path)
        base, ext = os.path.splitext(filename)
        csa_path = os.path.join(dirname, base + ".csa")
        os.makedirs(dirname, exist_ok=True)
        csa = CSA.Exporter(csa_path, encoding=args.encoding)

        csa.info(init_board=kif.sfen, names=kif.names)

        if len(kif.times) == 0:
            kif.times = [None] * len(kif.moves)
        for move, time, comment in zip(kif.moves, kif.times, kif.comments):
            csa.move(move, comment=remove_no_need_comment(comment, kif.names), time=time)

        csa.endgame(kif.endgame, time=kif.times[-1])
    except Exception as e:
        print(f"skip {path} {e}")
