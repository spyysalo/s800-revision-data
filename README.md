# S800 revision data

Data for revision of S800 corpus (https://species.jensenlab.org/)

## Original data

The original annotations of the S800 corpus were imported as follows:

```
wget http://species.jensenlab.org/files/S800-1.0.tar.gz
mkdir original-data
tar xzf S800-1.0.tar.gz -C original-data
```

The original annotations were then converted into standoff as follows:

```
git clone https://github.com/spyysalo/s800
cd s800
./convert_s800.sh ../original-data ../original-standoff
```

## Revised data

A copy of the original data in standoff format was imported as the
starting point for revisions

```
cp -r original-standoff revised-standoff
```

## PubMed texts

The PubMed XML data for each of the corpus documents was downloaded as
follows

```
mkdir pubmed-xml
for f in original-standoff/*.txt; do
    i=$(basename $f .txt)
    wget 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='"$i"'&rettype=abstract&retmode=xml' -O pubmed-xml/$i.xml
done
```

The PubMed text for each document was extracted as follows

```
mkdir pubmed-text
git clone https://github.com/spyysalo/pubmed.git
for f in pubmed-xml/*.xml; do
    python3 pubmed/extractTIABs.py -o - $f | egrep -v '$^' > pubmed-text/$(basename $f .xml).txt
done
```
