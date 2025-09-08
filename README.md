# LLM

#### This repository is to Optimize LLM performance using Multi-processing and Multi-threading in C Language.

## Multi-Processing

### File Structure

```bash
Multiprocessing
├── common.h # common and helper macro defns, read through first
├── main.c
├── inference.c # [your task] template for inference child process implementation
├── Makefile # makefile for the project
├── model.h # GPT model definition, modification not allowed
└── avg_cpu_use.py # Utility to parse the log and calculate average cpu usage
```

### Objective

    Optimization of Performance of LLM using Multiprocessing - Divide User Prompt Acception and Inference

### Model Description

- The LLM used is based on SmolLM by HuggingfaceTB.

- Llama3, an open-source variation of GPT, and complete single-thread LLM inference engine as the startpoint is provided.

- Inference framework used is based on the open-source project llama2.c by Andrej Karpathy.

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

#### Compile and run the inference program.

    make -B inference # -B:= make, force rebuild
    # or  manually
    gcc -o inference inference.c -o3 -lm

#### Please use -lm flag to link math library and -o3 flag to apply the best optimization allowed within C standard.

    ./main <seed> 2>log.txt
    # Put your prompt when >>> appears

#### Main process collects the running status of inference process.

#### All information about the statistics of a process are found using /proc/{pid}/stat and saved in log.txt

#### Informations are as follow:

| Item     | Description                                                |
| -------- | ---------------------------------------------------------- |
| pid      | Process ID                                                 |
| state    | Running Status                                             |
| Policy   | Scheduling Policy                                          |
| nice     | Nice value                                                 |
| vsize    | Virtual Memory Size                                        |
| task_cpu | CPI id of the process scheduled to                         |
| utime    | Running time of process spent in user mode, unit is 10ms   |
| stime    | Running time of process spent in system mode, unti is 10ms |

> please check [/proc/pid/stat manpage](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html) for more information

## Multi-Threading

### To Be Updated
