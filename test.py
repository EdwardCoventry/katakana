
import katakana

if __name__ == '__main__':

    # print(katakana.to_katakana('Edward'))
    print(katakana.to_katakana('Coventry'), '-', 'コヴェントリー', )

    print('Hello World', katakana.to_katakana('Hello World'.lower()), '-',  'ハロー・ワールド')
    print('Banana', katakana.to_katakana('Banana'.lower()), '-',  ' バナナ', )
    print('Test', katakana.to_katakana('Test'.lower()), '-',  'テスト', )
    print('Canada', katakana.to_katakana('Canada'.lower()), '-',  'カナダ', )
    print('Barbecue', katakana.to_katakana('Barbecue'.lower()), '-',  'バーベキュー')
    print('Google Maps', katakana.to_katakana('Google Maps'.lower()), '-', 'グーグル マップ')
    print('John Doe', katakana.to_katakana('John Doe'.lower()), '-',  'ジョン・ドウ')
    print('Donald Duck', katakana.to_katakana('Donald Duck'.lower()), '-',  'ドナルド・ダック')
    print('Donald Trump', katakana.to_katakana('Donald Trump'.lower()), '-',  'ドナルド・トランプ ')

    exit()
