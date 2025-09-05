/*
Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin
or with Makefile (recommended)
$ make prepare

Compile it with Level-3 Optimization and link math library (-lm, Linux built-in)
$ gcc -o inference inference_[UID].c -O3 -lm
or with Makefile (recommended)
$ make -B inference

Then run it with:
./inference <seed> <prompt1> <prompt2>
Example:
./inference 42 "What is Fibonacci Number?" "Can you give me a python program to generate Fibonacci Number?"
Note:
<prompt> must be quoted with ""
*/

#include "common.h"  // common headers

#include <unistd.h>  
#include <stdio.h>   // for printf, sprintf, fgets
#include <stdlib.h>  // for malloc, calloc
#include <stdint.h>  // for uint8_t and uint64_t
#include <string.h>  // for memcpy and strcmp

#include "model.h"// for LLM definitions -> no need to know

int pos = 0; // global 
Transformer transformer; // transformer instance to be init
Tokenizer tokenizer;     // tokenizer instance to be init
Sampler sampler;         // sampler instance to be init

// Define Additional Global Variables and Signal Handlers Here
#include <signal.h>
// Your Code Starts Here
// global variable
char buf[MAX_PROMPT_LEN]; // user prompt
sigset_t mask, old_mask;
void handle_SIGUSR1(int signum){
    // when signal is received, start to get user prompt from stdin
    //fprintf(stderr, "Child Process: SIGUSR1 Received\n");
 }

// Your Code Ends Here
// ----------------------------------------------------------------------------
// generation loop, don't modify

void generate(char *prompt) {
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+6) * sizeof(int)); // +6 reserved for prompt template
    encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;        // place holder for next token
    int token = prompt_tokens[0]; // place holder of prev token, kickoff as prompt_tokens[0]
    int end_pos = pos + MAX_NEW_TOKENS + num_prompt_tokens;
    int start_pos = pos;
    long start_time = 0; // to be lazy iniialzied
    while (pos < end_pos) {

        // forward the transformer to get logits for the next token
        float* logits = forward(&transformer, token, pos);

        if (pos < start_pos + num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos - start_pos + 1];
        } else if (pos == end_pos - 2) {
            // reaching the end, force it to close by <|im_end|>
            next = 2; // := <|im_end|>
        } else {
            // otherwise sample the next token from the logits
            next = sample(&sampler, logits);
        }

        pos++;

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(&tokenizer, token, next);
        if (pos >= num_prompt_tokens) {
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }

        token = next;

        // init the timer here because the first iteration can be slower
        if (start_time == 0) { start_time = time_in_ms(); }
    }
    printf("\n");

    long end_time = time_in_ms();
    printf("%d, %d, %ld, %ld \n", pos, start_pos, end_time, start_time);
    // \033[0;32m set color to green and \033[0m reset to default, they won't be generate by LLM
    fprintf(stdout, "\033[0;32m[INFO] SEQ LEN: %d at %.4f tok/s\033[0m\n", 
        pos, (pos - start_pos) / (float) (end_time - start_time) * 1000);
    
    free(prompt_tokens);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *model_path     = "model.bin";  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature    = 0.6f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp           = 0.9f;  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompts[4]     = {NULL, NULL, NULL, NULL};  // prompt strings
    int num_prompt       = 0; // number of prompts
    uint64_t rng_seed    = 0; // seed rng with time by default

    // parse command-line parameters via argv, you'll need to change this to read stdin
    // Your Code Starts Here
    struct sigaction SIGUSR1_handler;
    SIGUSR1_handler.sa_handler = handle_SIGUSR1;
    sigemptyset(&SIGUSR1_handler.sa_mask);
    SIGUSR1_handler.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &SIGUSR1_handler, NULL);
    // need to mask the SIGUSR1 here
    sigaddset(&mask, SIGUSR1);
    sigprocmask(SIG_BLOCK, &mask, NULL);
    /*
    if (argc >= 3) {
        rng_seed = atoi(argv[1]);
        num_prompt = (argc - 2) < 4 ? (argc - 2) : 4;
        for (int i = 0; i < num_prompt; i++) {
            prompts[i] = argv[i + 2];
        }
    }*/
    if(argc == 2){
        rng_seed = atoi(argv[1]);
        //fprintf(stderr, "%ld \n", rng_seed);
    }
    else {  /*
            fprintf(stderr, "Usage:   ./inference <seed> <prompt1> <prompt2>\n");
            fprintf(stderr, "Example: ./inference 42 \"What is Fibonacci Number?\" \"Can you give me a python program to generate Fibonacci Number?\"\n");
            fprintf(stderr, "Note:    <prompt> must be quoted with \"\", at most 4 prompt is supported \n");
            exit(EXIT_FAILURE);
            */
            // target interface
            fprintf(stderr, "Usage: ./inference <seed>\n");
            fprintf(stderr, "Note:  this shall not be called directly, use ./entry <seed> \n");
            exit(EXIT_FAILURE);
        }

    prompts[num_prompt] = buf;
    // Your Code Ends Here

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    // build the Transformer via the model .bin file
    build_transformer(&transformer, model_path);
    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // Generation Loop, update to match requirements
    // Your code starts here
        /*
        for (int i = 0; i < num_prompt; i++) {
            printf("user\n %s \n", prompts[i]);
            generate(prompts[i]);   
        }*/

    while(num_prompt < 4){
        sigsuspend(&old_mask);
        sigprocmask(SIG_UNBLOCK, &mask, NULL);
        if(fgets(buf, MAX_PROMPT_LEN, stdin) == NULL){fprintf(stderr, "Prompt reading Error in child process");} 
        prompts[num_prompt] = buf;   
        generate(prompts[num_prompt]);
        num_prompt++;  
        kill(getppid(), SIGUSR2);
        sigprocmask(SIG_BLOCK, &mask, NULL);
    }
    // Your code ends here
    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    exit(0);
}










