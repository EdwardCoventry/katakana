# English-Katakana Names

An English-Katakana names dataset aggregated from Wikipedia's article titles.


### How to build the dataset

From [DBPedia's website](https://wiki.dbpedia.org/downloads-2016-10) download "Labels" dataset in English and Japanese. The latest available data is 2016-10.

```
# Use 2016-10 version
wget http://downloads.dbpedia.org/2016-10/core-i18n/en/labels_wkd_uris_en.ttl.bz2
wget http://downloads.dbpedia.org/2016-10/core-i18n/ja/labels_wkd_uris_ja.ttl.bz2
bzip2 -d *.bz2
```

The `build_dataset.py` works with TTL format.

After download and unarchived, you should have `labels_wkd_uris_en.ttl` and `labels_wkd_uris_ja.ttl`. Their content should follow pattern `<url> <schema> "title"@(en|jp)`.

``` 
<http://wikidata.dbpedia.org/resource/Q1000013> <http://www.w3.org/2000/01/rdf-schema#label> "ジャガー・XK140"@ja .
<http://wikidata.dbpedia.org/resource/Q1000022> <http://www.w3.org/2000/01/rdf-schema#label> "寧波軌道交通"@ja .
<http://wikidata.dbpedia.org/resource/Q1000032> <http://www.w3.org/2000/01/rdf-schema#label> "アンスクーリング"@ja .
...

<http://wikidata.dbpedia.org/resource/Q1000000> <http://www.w3.org/2000/01/rdf-schema#label> "Water crisis in Iran"@en .
<http://wikidata.dbpedia.org/resource/Q1000001> <http://www.w3.org/2000/01/rdf-schema#label> "Gold Cobra"@en .
<http://wikidata.dbpedia.org/resource/Q1000003> <http://www.w3.org/2000/01/rdf-schema#label> "Nielles-lès-Bléquin"@en .
...
```

Then, run `build_dataset.py`. The script will
- Parse and extract (url, title) pairs from `labels_wkd_uris_en.ttl` to `english_titles.csv`.
- Parse and extract (url, title) pairs from `labels_wkd_uris_en.ttl` to `english_titles.csv`.
- Find Katakana only Japanese titles
- Join with English title by the resouce link
- Write the join title pairs into `data.csv`

```
./build_dataset.py
```
