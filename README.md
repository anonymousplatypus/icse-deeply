# Our Approach [Modified CO-GCN]

## Instructions to run

* Install dependencies: `pip3 install requirements.txt`.
* In the same way as all the other approaches, add the datasets to `./datasets_runtime/`.
* Add the weights you want to use on L27 of `cogcn/cogcn.py`. If the weights are unknown, set it to an empty dictionary. Line 27 has several examples of this with known weights. For unknown weights, this would simply be:
```{py}
'application': {}
```
* Run the code: `cd cogcn && python3 cogcn.py`.


## Output

The output will be on the terminal. You can redirect it as desired. For example, in an unsupervised VM such as Google Cloud, you might use

```{sh}
nohup python3 cogcn.py > output.txt 2>&1 &
```

Currently, the program outputs a tab-separated output. You could set it to other formats in L206. Examples are CSV (`tablefmt='csv'`), GitHub-flavored Markdown tables (`tablefmt='github'`).
