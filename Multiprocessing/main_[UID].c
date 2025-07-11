/*
* PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: 
* NAME: 
* UID:  
* Development Platform: 
* Remark: (How much you implemented?)
* How to compile separately: (gcc -o main main_[UID].c)
*/

#include "common.h"  // common definitions

#include <stdio.h>   // for printf, fgets, scanf, perror
#include <stdlib.h>  // for exit() related
#include <unistd.h>  // for folk, exec...
#include <wait.h>    // for waitpid
#include <signal.h>  // for signal handlers and kill
#include <string.h>  // for string related 
#include <sched.h>   // for sched-related
#include <syscall.h> // for syscall interface

#define READ_END       0    // helper macro to make pipe end clear
#define WRITE_END      1    // helper macro to make pipe end clear
#define SYSCALL_FLAG   0    // flags used in syscall, set it to default 0

// Define Global Variable, Additional Header, and Functions Here
#include <errno.h>
int got_signal = 0;
// user prompt input
char buf[MAX_PROMPT_LEN];

int main(int argc, char *argv[]) {
    char* seed; // 
    if (argc == 2) {
        seed = argv[1];
    } else if (argc == 1) {
        // use 42, the answer to life the universe and everything, as default
        seed = "42";
    } else {
        fprintf(stderr, "Usage: ./main <seed>\n");
        fprintf(stderr, "Note:  default seed is 42\n");
        exit(1);
    }
    // option 1: main program = parent process & inference = child process
    // option 2: inference = parent process & main program = child process

    // Write your main logic here
    pid_t process;
    int pfd[2];
    pipe(pfd);
    // use fork to create child process 
    process = fork();

    if (process == 0){ // use exec to run inference_[UID].c for child process 
        close(pfd[WRITE_END]); // close the write end for child process
        if (dup2(pfd[READ_END], READ_END) == -1){ // set pipe read end to stdin
            fprintf(stderr, 'dup2 failed');} 
        if (execlp("inference", "inference", seed) == -1){
            fprintf(stderr, "execlp: error no = %s\n", strerror(errno));}
    } else { // in the main process, accept user input
        close(pfd[READ_END]); // close read end for main process
        // accept the user prompt up to 4 or until the SIGINT is received
        while(fgets(buf, MAX_PROMPT_LEN, stdin)){}
        size_t len = strlen(buf);
        // pass the prompt to the pipeline to the child process.
        if (write(pfd[WRITE_END], buf, len) == -1){
            fprintf(stderr, "Main process failed to write on child child process.", stderror(errno))
        }
        // send the SIGUSR1 to child process
        kill(SIGUSR1);
        // while the child process is inferencing
        while(!got_signal){
            //stop accept new prompt input
        }
    }
    

    
    return EXIT_SUCCESS;
}