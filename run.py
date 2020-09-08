#!/usr/bin/env python
import os

## extract features

# from all the good files
for i in [j for j in os.listdir() if j.endswith('wav')]:
    print(i)
