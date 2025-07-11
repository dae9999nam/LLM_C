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
// install signal handler here
int child_done, sigint_received = SYSCALL_FLAG; // child is not done inferencing
void sigint_handler(int sig){
    sigint_received = 1;
}
void sigusr2_handler(int sig){
    child_done = 1;
}

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
    // Write your main logic here
    pid_t pid;
    int pfd[2]; // pipe file descriptor
    pipe(pfd);
    // install signal handlers here
    struct sigaction sa = {0};
    sa.sa_handler = sigusr2_handler;
    sigaction(SIGUSR2, &sa, NULL);
    sa.sa_handler = sigint_handler;
    sigaction(SIGINT, &sa, NULL);

    //block SIGUSR2 and SIGINT so they queue up
    sigset_t mask, oldmask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR2);
    sigaddset(&mask, SIGINT);
    if (sigprocmask(SIG_BLOCK, &mask, &oldmask)< 0){fprintf(stderr, "Sigprocmask failed error no = %s \n", stderr(errno))}

    // use fork to create child process 
    pid = fork();
    if (pid == 0){
        close(pfd[WRITE_END]); // close the write end for child process
        if (dup2(pfd[READ_END], READ_END) == -1){ // set pipe read end to stdin
            fprintf(stderr, "dup2 failed: %s\n", strerror(errno));} 
        close(pfd[READ_END]); // close the pipe read end
        if (execlp("inference", "inference", seed) == -1){ // use exec to run inference_[UID].c for child process 
            fprintf(stderr, "execlp: error no = %s\n", strerror(errno));}
    } 
    // main process: 
    // get user prompt -> pass to inference process
    // -> wait until the inference process finish inferencing & access to /proc to retrieve cpu usage information
    else { // in the main process, accept user input
        char buf[MAX_PROMPT_LEN];

        close(pfd[READ_END]); // close read end for main process
        // accept the user prompt up to 4 or until the SIGINT is received
        for(int i = 0; i < 4 && !sigint_received; i++){ // run until SIGINT not received or num_prompt < 4
            if(!fgets(buf, MAX_PROMPT_LEN, stdin)){break;}
            // pass the prompt to the pipeline to the child process.
            if (write(pfd[WRITE_END], buf, strlen(buf)) == -1){
                fprintf(stderr, "Main process failed to write on child child process = error no %s\n", stderror(errno))
            }
            kill(pid, SIGUSR1); // send SIGUSR1 to child process to notice the user prompt is ready
            // while the child process is inferencing
            while(!child_done && !sigint_received) sigsuspend(&oldmask);
            child_done = SYSCALL_FLAG; // reset the flag
        }   
        // close the write end
        close(pfd[WRITE_END]);
        // restore the original mask if
        sigprocmask(SIG_SETMASK, &oldmask, NULL);
        wait(NULL); // wait for child process to finish
    }
    

    
    return EXIT_SUCCESS;
}