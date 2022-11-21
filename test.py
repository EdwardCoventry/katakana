
import katakana

if __name__ == '__main__':

    # print(katakana.to_katakana('Edward'))
    print(katakana.to_katakana('Coventry'), '-', 'コヴェントリー', )

    print('Hello World', katakana.to_katakana('Hello World'.lower()), '-',  'ヘロ・ウォルド', 'ハロー・ワールド')
    print('Banana', katakana.to_katakana('Banana'.lower()), '-',  ' バナナ', 'バナナ')
    print('Test', katakana.to_katakana('Test'.lower()), '-',  'テスト', '')
    print('Canada', katakana.to_katakana('Canada'.lower()), '-',  'カナダ', '')
    print('Barbecue', katakana.to_katakana('Barbecue'.lower()), '-',  'バルベク', 'バーベキュー')
    print('Google Maps', katakana.to_katakana('Google Maps'.lower()), '-',  'グードル・マップス', 'グーグル マップ')
    print('John Doe', katakana.to_katakana('John Doe'.lower()), '-',  'ジョン・デイ', 'ジョン・ドウ')
    print('Donald Duck', katakana.to_katakana('Donald Duck'.lower()), '-',  'ドナルド・ダック', 'ドナルド・ダック')
    print('Donald Trump', katakana.to_katakana('Donald Trump'.lower()), '-',  'ドナルド・トルプング', 'ドナルド・トランプ ')

    exit()
