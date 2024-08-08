import katakana


def print_katakana(name, katakana_name):
    transliterated_katakana = katakana.to_katakana(
        name.lower(),
        use_tflite=True,

    )

    print(f"{name} -> {transliterated_katakana}    ({katakana_name})")


if __name__ == '__main__':
    test_cases = [
        ('Coventry', 'コヴェントリー'),
        ('Hello・World', 'ハロー・ワールド'),
        ('Banana', 'バナナ'),
        ('Test', 'テスト'),
        ('Canada', 'カナダ'),
        ('Barbecue', 'バーベキュー'),
        ('Google・Maps', 'グーグル マップ'),
        ('John・Doe', 'ジョン・ドウ'),
        ('Donald・Duck', 'ドナルド・ダック'),
        ('Donald・Trump', 'ドナルド・トランプ')
    ]

    for name, katakana_name in test_cases:
        print_katakana(name, katakana_name)

    exit()
