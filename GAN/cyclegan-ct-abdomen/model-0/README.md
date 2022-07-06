

+ train

```
cd model-0
# create/update submit.condor and relevant sh file
condor_submit submit.condor
```


+ monitor

```

tensorboard --port=6009 --bind_all --logdir=log 
```