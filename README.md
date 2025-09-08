# LLM

#### This repository is to Optimize LLM performance using Multi-processing and Multi-threading in C Language.

## Multi-Processing

### Obejective

        Optimization of Performance of LLM using Multiprocessing - Divide User Prompt Acception and Inference

### Model Description

##### The LLM used is based on SmolLM by HuggingfaceTB.

##### Llama3, an open-source variation of GPT, and complete single-thread LLM inference engine as the startpoint is provided.

##### the inference framework used is based on the open-source project llama2.c by Andrej Karpathy.

### How it works

##### Please download the model and tokenizer to the same folder:

        $ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
        $ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin
        or with Makefile (recommended)
        $ make prepare

##### Compile it with Level-3 Optimization and link math library (-lm, Linux built-in)

        $ gcc -o inference inference.c -O3 -lm
        or with Makefile (recommended)
        $ make -B inference

### Compile and run the inference program.

    make -B inference # -B:= make, force rebuild
    # or  manually
    gcc -o inference inference.c -o3 -lm

#### Please use -lm flag to link math library and -o3 flag to apply the best optimization allowed within C standard.

    ./main <seed> 2>log.txt
    # Put your prompt when >>> appears
    # While Inferencing, log.txt file will be generated
    # In the log.txt, there are
    # "[pid] [tcomm] [state] [policy] [nice] [vsize] [task_cpu] [utime] [stime] [cpu%] "

## Multi-Threading

### To Be Updated
