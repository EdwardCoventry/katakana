import katakana


def print_katakana(name, katakana_name):
    transliterated_katakana = katakana.to_katakana(
        name.lower(),
        use_tflite=False,

    )

    print(f"{name} -> {transliterated_katakana}    ({katakana_name})")


if __name__ == '__main__':
    test_cases = {
        'number9isanumberwichisalsoknownas9': 'ナンバー9イズアナンバーウィッチイズアルソウノウナズ9',
        'supercalifragilisticexpialidocious': 'スーパーキャリフラジリスティックエクスピアリドーシャス',
        'nitrobenzenesulfenanilide': 'ニトロベンゼンスルフェナニリド',
        'alsok': 'アルソック',
        'orkest': 'オーケスト',
        'reds': 'レッズ',
        'vegetabrella': 'ベジタブレラ',
        'palimpalim': 'パリンパリン',
        'palitextdestroy': 'パリテキストデストロイ',
        'pandespiegle': 'パンデスピーグル',
        'Coventry': 'コヴェントリー',
        'Hello・World': 'ハロー・ワールド',
        'Banana': 'バナナ',
        'Test': 'テスト',
        'Canada': 'カナダ',
        'Barbecue': 'バーベキュー',
        'Google・Maps': 'グーグル・マップ',
        'John・Doe': 'ジョン・ドウ',
        'Donald・Duck': 'ドナルド・ダック',
        'Donald・Trump': 'ドナルド・トランプ',
        'adio': 'アディオ',
        'sc': 'スク',
        'Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch': 'ランヴェアプルグウィングルゴゲリュフウィルンドロブウリュランティスィリオゴゴゴホ'
    }
    for name, katakana_name in test_cases.items():
        print_katakana(name, katakana_name)

    exit()