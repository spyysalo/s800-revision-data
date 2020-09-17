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
