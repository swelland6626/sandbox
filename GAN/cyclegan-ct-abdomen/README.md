


```

materials used for dicussion with Spencer on his gan project - 2022-03-23
```

+ build container - 5min

```
bash build.sh

export REMOTE_URL=registry.cvib.ucla.edu/$USER:cyclegan-ct-abdomen
docker tag cyclegan-ct-abdomen $REMOTE_URL
docker push $REMOTE_URL


docker run -it -v $PWD:/workdir -w /workdir -v /radraid:/radraid -u $(id -u):$(id -g) -p 7009:7009 cyclegan-ct-abdomen bash

docker run -it -v $PWD:/workdir -w /workdir -v /radraid:/radraid -u $(id -u):$(id -g) -p 6007:6007 cyclegan-ct-abdomen bash

docker run -it -v $PWD:/workdir -w /workdir -v /radraid:/radraid -u $(id -u):$(id -g) -p 6008:6008 cyclegan-ct-abdomen bash


```

+ image download from tcia (contrast, no-contrast) 20 min.

```
# python download.py sample.csv /radraid
# api not functioning today https://github.com/TCIA-Community/TCIA-API-SDK/pull/1#issuecomment-1076581509
# instead used java app, `NBIA Data Retriever`

```

+ create csv file to seperate phases.

```
cd prepare
python prepare.py /radraid/pteng/tmp/c4kc-kits

```

+ train using published/available cyclegan. see `model-0`

