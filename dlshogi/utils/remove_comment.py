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
    # 場所
    "千駄ヶ谷", "連盟", "会館","ホテル","レストラン","旅館","温泉","スタジオ","記念館","庭園", "式場", "対局場", 
    "北海道","青森","岩手","宮城","秋田","山形","福島","茨城","栃木","群馬","埼玉","千葉","東京","神奈川",
    "新潟","富山","石川","福井","山梨","長野","岐阜","静岡","愛知","三重","滋賀","京都","大阪","兵庫","奈良","和歌山",
    "鳥取","島根","岡山","広島","山口","徳島","香川","愛媛","高知","福岡","佐賀","長崎","熊本","大分","宮崎","鹿児島","沖縄","甲府",
    # 食べ物
    "食事","メニュー","注文","昼食", "夕食", "おやつ", "軽食","デザート","スイーツ","うどん","そば","弁当","寿司","ラーメン","カレー","定食","牛丼","天ぷら","焼肉","ステーキ","ハンバーグ",
    "サンドイッチ","ケーキ","アイス","ジュース","コーヒー","紅茶","ピザ","パスタ","オムライス","カツ丼","鰻","うな重",
    "餃子","唐揚げ","刺身","グラタン","ティー", "プリン", "チョコレート",
    # 時間帯
    "梅雨",
    # その他対局外
    "開始","終了","休憩","感想戦","再開","立会","記録係","ABEMA","ニコ生","中継","テレビ",
    "出身","生まれ","党","棋士番号","永世", "芙蓉","水無瀬","錦旗", "○", "●", "女流", "アマチュア", "YouTube", "=", "＝","菱湖書", "モバイル",
    "kifulog", "本戦", "予選", "観光", "タイトル保持者", "羽織", "呉服", "着物", "記事",
    "http", "回", "名人", "第", "段", "級", "本局", "BS", "自動棋譜更新",
    "消費時間", "コメント", "来訪", "【", "受賞", "時", "◆", "本日",
    "対局数", "感想戦取材", "】", "敗",
    "期", "分", "時刻", "レスポンシブ", "気温", "奨励会",
    "局後の感想", "スマートフォン", "棋戦", "優勝", "勝", "成績", "入室",
    "※", "タブレット", "将棋会館", "腕組み", "対戦", "席",  
    "天気", "■", "AbemaTV", "Twitter", "HP", "菓子", "live_id", "料金", "執筆", "大盤解説","channel", "会員", "会社", "リンク先", "カメラ"
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
    out = "".join(filtered_comments)
    if out == "":
        out = None
    return out

# if __name__ == "__main__":
#     print(remove_no_need_comment(comments))
