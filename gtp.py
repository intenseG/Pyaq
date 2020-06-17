#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必要なライブラリ群をインポート
from sys import stderr, stdout, stdin
import numpy as np
from board import *
from search import Tree

# 対応しているGTPコマンドのリスト
cmd_list = [
    "protocol_version", "name", "version", "list_commands",
    "boardsize", "komi", "time_settings", "time_left",
    "clear_board", "genmove", "play", "undo",
    "gogui-play_sequence", "showboard", "quit"
]


# strにcmdが含まれていればTrue、そうでなければFalseを返す汎用関数
def include(str, cmd):
    return str.find(cmd) >= 0


# GTPコマンドの処理結果を標準出力に出力する関数
def send(res_cmd):
    # GTPコマンドの処理結果を標準出力に書き込む
    stdout.write("= " + res_cmd + "\n\n")
    stdout.flush()


# 空白区切りの文字列を空白で分割してリストで返す関数
def args(str):
    # 空白区切りの文字列を空白でsplit関数で分割してリスト化する
    # 例: 第1引数strの中身が = 300 15 3 の場合、=, 300, 15, 3 の4つの要素を含むリストを返す
    arg_list = str.split()
    # リストの先頭の要素が = であればpop関数で0番目の要素を削除する
    if arg_list[0] == "=":
        arg_list.pop(0)
    arg_list.pop(0)
    return arg_list

# GTPコマンドを常に受け付けるメインループ
def call_gtp(main_time, byoyomi, quick=False, clean=False, use_gpu=True):
    # board.pyのBoardクラスをインスタンス化する
    b = Board()
    # search.pyのTreeクラスをインスタンス化する
    tree = Tree(use_gpu=use_gpu)
    # 引数で受け取ったmain_timeとbyoyomiをTreeクラスの変数に代入する
    tree.main_time = main_time
    tree.byoyomi = byoyomi

    # メインループ開始
    while 1:
        # 標準入力を読み取って、rstrip関数で改行コードを除去する
        # これはプログラムが終了するまで実行され続ける
        str = stdin.readline().rstrip("\r\n")
        # strが空だったらcontinueする
        if str == "":
            continue
        Tree.stop = True
        ### プロトコルバージョン
        # 入力例: protocol_version
        if include(str, "protocol_version"):
            send("2")
        ### プログラム名
        # 入力例: name
        elif include(str, "name"):
            send("Pyaq")
        ### プログラムバージョン
        # 入力例: version
        elif include(str, "version"):
            send("1.0")
        ### GTPコマンドリスト
        # 入力例: list_commands
        elif include(str, "list_commands"):
            stdout.write("=")
            for cmd in cmd_list:
                stdout.write(cmd + "\n")
            stdout.write("\n")
            stdout.flush()
        ### 碁盤サイズ(board.pyのBSIZE変数の値と一致する必要あり)
        # コマンド引数: 碁盤サイズ
        # 入力例: boardsize 9
        elif include(str, "boardsize"):
            bs = int(args(str)[0])
            if bs != BSIZE:
                stdout.write("?invalid boardsize\n\n")
            send("")
        ### コミ
        # コマンド引数: コミ値
        # 入力例: komi 7.5
        elif include(str, "komi"):
            send("")
        ### 時間設定
        # コマンド引数: 持ち時間, 秒読み, 秒読み回数
        # 入力例: time_settings 300 15 3
        elif include(str, "time_settings"):
            arg_list = args(str)
            tree.main_time = arg_list[0]
            tree.left_time = tree.main_time
            tree.byoyomi = arg_list[1]
        ### 残り時間
        # コマンド引数: 手番(b or w), 残り時間, 残り時間回数
        # 入力例: time_left b 150 1
        elif include(str, "time_left"):
            tree.left_time = float(args(str)[1])
        ### 盤面をリセット
        # 入力例: clear_board
        elif include(str, "clear_board"):
            b.clear()
            tree.clear()
            send("")
        ### AIに思考させて着手を生成
        # コマンド引数: 手番(b or w)
        # 入力例: genmove b
        elif include(str, "genmove"):
            # 起動時にquickモードを指定した場合は探索せずに評価最大の手を選択する
            if quick:
                win_rate = 0.5
                move = rv2ev(np.argmax(tree.evaluate(b)[0][0]))
            # quickモードでなければ探索して、着手と勝率をtuple型で返す
            else:
                move, win_rate = tree.search(b, 0, ponder=False, clean=clean)

            # winrate(勝率)が10%未満だったら投了する
            if win_rate < 0.1:
                send("resign")
            # winrate(勝率)が10%以上なら着手する
            else:
                b.play(move)
                send(ev2str(move))
        ### AIに思考させず盤面に着手
        # コマンド引数: 手番(b or w), 座標
        # 入力例: play b Q16
        elif include(str, "play"):
            b.play(str2ev(args(str)[1]), not_fill_eye=False)
            send("")
        ### マッタ(1手戻す)
        # 入力例: undo
        elif include(str, "undo"):
            history = b.history[:-1]
            b.clear()
            tree.clear()
            for v in history:
                b.play(v, not_fill_eye=False)

            send("")
        ### [gogui専用コマンド] 複数の手の履歴をまとめて盤面に反映
        # コマンド引数: 座標(x手目), 座標(x + 1手目), 座標(x + 2手目)...
        # 入力例: gogui-play_sequence Q16 D4 Q4 D16 K10
        elif include(str, "gogui-play_sequence"):
            arg_list = args(str)
            for i in range(1, len(arg_list) + 1, 2):
                b.play(str2ev(arg_list[i]), not_fill_eye=False)
            send("")
        ### 現在の盤面をテキストで表示
        # 入力例: showboard
        elif include(str, "showboard"):
            b.showboard()
            send("")
        ### プログラム終了
        # 入力例: quit
        elif include(str, "quit"):
            send("")
            break
        ### プログラム対応していないGTPコマンド
        else:
            stdout.write("?unknown_command\n\n")
