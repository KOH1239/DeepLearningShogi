import argparse
import cshogi
import cshogi.KIF

parser = argparse.ArgumentParser()
parser.add_argument("comment")
args = parser.parse_args()

# ja_sentence_segmenter など必要なライブラリのインポート
import functools
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

split_punc2 = functools.partial(split_punctuation, punctuations=r"。!?")
concat_tail_te = functools.partial(concatenate_matching, former_matching_rule=r"^(?P<result>.+)(て)$", remove_former_matched=False)
segmenter = make_pipeline(normalize, split_newline, concat_tail_te, split_punc2)

path = "/home/koh2357/kif_comment/A_ryuou_kif/ryuou201206290101.kif"
kif = cshogi.KIF.Parser.parse_file(path)

comments = args.comment

comments ="後手番となった永瀬の作戦は、角道を止めない四間飛車。以前は先手で早石田、後手でゴキゲン中飛車を中心に戦っていた永瀬だが、最近は先後問わず様々な場所に飛車を振る。球種が多ければ多いほど、相手は的を絞りにくくなる。"
# 指定されたキーワードのリスト
keywords = [
    "http", "昼食", "夕食", "回", "名人", "第", "段", "本局", "BS", "自動棋譜更新", 
    "消費時間", "コメント", "来訪", "【", "受賞", "出身", "党", "時", "◆", "本日", 
    "対局数", "ネット中継", "記録係", "開始", "感想戦取材", "再開", "】", "敗", 
    "期", "棋士番号", "分", "時刻", "立会", "永世", "レスポンシブ", "気温", "奨励会", 
    "局後の感想", "スマートフォン", "おやつ", "棋戦", "優勝", "勝", "成績", "入室", 
    "ABEMA", "※", "タブレット", "生まれ", "将棋会館", "腕組み", "対戦", "席", "食事", 
    "天気", "■"
]

def remove_no_need_comment(comments):
    comments = list(segmenter(comments))
    filtered_comments = [
            comment for comment in comments 
            if comment is not None and not any(keyword in comment for keyword in keywords)
            ]
    
    return filtered_comments

if __name__ == "__main__":
    # print(remove_no_need_comment(comments))
    names = ['大石直嗣', '永瀬拓矢']
    sentences = [
        '後手番となった永瀬の作戦は、角道を止めない四間飛車。',
        '以前は先手で早石田、後手でゴキゲン中飛車を中心に戦っていた永瀬だが、最近は先後問わず様々な場所に飛車を振る。',
        '球種が多ければ多いほど、相手は的を絞りにくくなる。'
    ]

    # 名前を部分一致で置き換える処理
    for i, sentence in enumerate(sentences):
        print(f"Before change (sentence {i}): {sentence}")  # 変更前を表示
        for name in names:
            if name.split(' ')[0] in sentence:  # 部分一致を許可
                if i == 0:  # 0番目の文は「先手」
                    sentence = sentence.replace(name.split(' ')[0], '先手')
                elif i == 1:  # 1番目の文は「後手」
                    sentence = sentence.replace(name.split(' ')[0], '後手')
        print(f"After change (sentence {i}): {sentence}")  # 変更後を表示
        sentences[i] = sentence

    print("\nFinal sentences:")
    print(sentences)
