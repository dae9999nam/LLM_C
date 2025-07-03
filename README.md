# LLM

#### This repository is to optimize LLM interface and performance using Multi-processing and Multi-threading in C Language.

#### Later on, Research in LLM in Python will be added.

#### Obejective

    To have hands-on practice in designing and developing a chatbot program, which involves the creation, management and coordination of processes.

## Multi-Processing

#### Llama3, an open-source variation of GPT, and complete single-thread LLM inference engine as the startpoint is provided.

##### the inference framework used is based on the open-source project llama2.c by Andrej Karpathy.

##### the LLM used is based on SmolLM by HuggingfaceTB.

### Compile and run the inference program.

    make -B inference # -B:= make, force rebuild
    # or  manually
    gcc -o inference inference_[UID].c -o3 -lm

#### Please use -lm flag to link math library and -o3 flag to apply the best optimization allowed within C standard.

    ./inference <seed> "<prompt>" "<prompt>" # prompt must quoted with ""
    # examples
    ./inference 42 "What is Fibonacci number?"

## Multi-Threading
