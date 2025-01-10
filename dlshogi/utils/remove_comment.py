import argparse
import cshogi
import cshogi.KIF

# ja_sentence_segmenter など必要なライブラリのインポート
import functools
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation
import re

split_punc2 = functools.partial(split_punctuation, punctuations=r"。!?")
concat_tail_te = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(て)$", remove_former_matched=False)
segmenter = make_pipeline(normalize, split_newline, concat_tail_te, split_punc2)

# path = "/home/koh2357/kif_comment/A_ryuou_kif/ryuou201206290101.kif"
# kif = cshogi.KIF.Parser.parse_file(path)

# comments ="後手番となった永瀬の作戦は、角道を止めない四間飛車。以前は先手で早石田、後手でゴキゲン中飛車を中心に戦っていた永瀬だが、最近は先後問わず様々な場所に飛車を振る。球種が多ければ多いほど、相手は的を絞りにくくなる。"
# 指定されたキーワードのリスト
keywords = [
    "http", "昼食", "夕食", "回", "名人", "第", "段", "本局", "BS", "自動棋譜更新", 
    "消費時間", "コメント", "来訪", "【", "受賞", "出身", "党", "時", "◆", "本日", 
    "対局数", "ネット中継", "記録係", "開始", "感想戦取材", "再開", "】", "敗", 
    "期", "棋士番号", "分", "時刻", "立会", "永世", "レスポンシブ", "気温", "奨励会", 
    "局後の感想", "スマートフォン", "おやつ", "棋戦", "優勝", "勝", "成績", "入室", 
    "ABEMA", "※", "タブレット", "生まれ", "将棋会館", "腕組み", "対戦", "席", "食事", 
    "天気", "■", "AbemaTV", "Twitter", "HP", "菓子", "live_id", "料金", "執筆", "大盤解説"
]

def remove_no_need_comment(comments, names):
    # 着手に記載されたコメントの中で、指定したキーワードの入った文を削除する
    comments = list(segmenter(comments))
    filtered_comments = [
            comment for comment in comments 
            if comment is not None and not any(keyword in comment for keyword in keywords)
            ]
    
    if any(any(name[:2] in comment for name in names) for comment in filtered_comments):    
        for i, sentence in enumerate(filtered_comments):
            for name in names:
                if name[:2] in sentence:
                    if name == names[0]:
                        sentence = re.sub(fr"{name[:2]}.*?", "先手", sentence)
                    elif name == names[1]:
                        sentence = re.sub(fr"{name[:2]}.*?", "後手", sentence)
            filtered_comments[i] = sentence
    
    return "".join(filtered_comments)

# if __name__ == "__main__":
#     print(remove_no_need_comment(comments))
