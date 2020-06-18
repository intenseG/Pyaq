#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 必要なライブラリ群をインポート
from collections import Counter
import sys
from board import *
import gtp
import learn
import search

# ここが一番最初に実行される
if __name__ == "__main__":
    # プログラム実行時コマンドラインに指定した引数を受け取る
    args = sys.argv

    ### 起動モード
    # 0: GTP対戦, 1: 自己対戦, 2: 教師あり学習
    launch_mode = 0
    ### 秒読み
    byoyomi = 5.0
    ### 持ち時間
    main_time = 0.0
    ### 確率最大の手を選択するモード(探索なし)
    quick = False
    ### ランダムに着手するモード(探索なし)
    random = False
    ### 最後まで打ち切るモード(探索ありの場合のみ)
    clean = False
    ### GPUを使うかどうか
    # True: 使う, False: 使わない
    use_gpu = True

    # 引数リストをループで回して起動モードや制限時間などの設定をする
    for arg in args:
        # 自己対戦モードON
        if arg.find("self") >= 0:
            launch_mode = 1
        # 教師あり学習モードON
        elif arg.find("learn") >= 0:
            launch_mode = 2
        # 確率最大の手を選択するモードON
        elif arg.find("quick") >= 0:
            quick = True
        # ランダムに着手するモードON
        elif arg.find("random") >= 0:
            random = True
        # 最後まで打ち切るモードON
        elif arg.find("clean") >= 0:
            clean = True
        # 持ち時間を設定
        elif arg.find("main_time") >= 0:
            main_time = float(arg[arg.find("=") + 1:])
        # 秒読みを設定
        elif arg.find("byoyomi") >= 0:
            byoyomi = float(arg[arg.find("=") + 1:])
        # CPUのみを使用する(GPUモードOFF)
        elif arg.find("cpu") >= 0:
            use_gpu = False

    # GTPモード
    if launch_mode == 0:
        # GTPコマンドを常に受け付けるメインループ開始
        gtp.call_gtp(main_time, byoyomi, quick, clean, use_gpu)

    # 自己対戦モード
    elif launch_mode == 1:
        # board.pyのBoardクラスをインスタンス化する
        b = Board()
        # ランダムに着手するモードでなければ
        if not random:
            # search.pyのTreeクラスを学習済みモデルを読み込んでインスタンス化する
            tree = search.Tree("model.ckpt", use_gpu)

        # 自己対戦のメインループ開始
        # 総手数が 碁盤サイズ * 碁盤サイズ * 2 以上になったらループ強制終了
        # 例: 19 * 19 * 2 = 722手
        while b.move_cnt < BVCNT * 2:
            # 1手前の手を保持(連続パスで終局するため)
            prev_move = b.prev_move
            # ランダムに着手するモードだったら
            if random:
                # 着手可能な座標リストからランダムに選択し着手する
                move = b.random_play()
            # 確率最大の手を選択するモードだったら
            elif quick:
                # 学習済みモデルから確率最大の手を選択する
                move = rv2ev(np.argmax(tree.evaluate(b)[0][0]))
                # 盤面に石を打つ
                b.play(move, False)
            # それ以外だったら
            else:
                # モンテカルロ木探索で探索して最善の手を選択する
                move, _ = tree.search(b, 0, clean=clean)
                # 盤面に石を打つ
                b.play(move, False)

            # 盤面をテキストで表示する
            b.showboard()
            # 連続でパスした場合、ループを抜ける
            if prev_move == PASS and move == PASS:
                break

        # 陣地の計算結果を入れるリスト
        score_list = []
        # 現在のBoardクラスのコピーを入れるためのインスタンス
        b_cpy = Board()

        # ロールアウトで打てるところが無くなるまで高速プレイを256回繰り返す
        for i in range(256):
            # 現在のBoardクラスをb_cpy変数にコピーする
            b.copy(b_cpy)
            # 打てるところが無くなるまで高速プレイする
            b_cpy.rollout(show_board=False)
            # 陣地の計算結果をリストに追加する
            score_list.append(b_cpy.score())

        # 陣地の計算結果が256回分入ったリストから、最も出現回数の多いものを取り出す
        score = Counter(score_list).most_common(1)[0][0]
        # 白と黒の陣地の差が0だったら
        if score == 0:
            # 引き分け
            result_str = "Draw"
        # 白と黒の陣地の差が0でなければ
        else:
            # scoreの値が0より大きければ黒勝ち、0より小さかったら白勝ち
            winner = "B" if score > 0 else "W"
            # 結果表示用の文字列型変数
            result_str = "%s+%.1f" % (winner, abs(score))
        # 標準エラー出力に対局結果を出力
        sys.stderr.write("result: %s\n" % result_str)

    # 教師あり学習モード
    else:
        # 教師あり学習開始
        # 第1引数: 学習率 -> 3e-4(0.0003)
        # 第2引数: ドロップアウト率 -> 0.5
        # 第3引数: SGFファイルのあるディレクトリパス -> sgf/
        # 第4引数: 学習にGPUを使うかどうか -> True
        # 第5引数: GPUの枚数 -> 1
        learn.learn(3e-4, 0.5, sgf_dir="sgf/", use_gpu=use_gpu, gpu_cnt=1)
