/*
* FILE NAME: 
* NAME: BAEK SEUNGHEYON
* Development Platform: Visual Studio Code & Docker
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
#include <sys/resource.h>
// file path for /proc/pid
char path[256]; 
// install signal handler here
int SIGUSR2_flag = SYSCALL_FLAG; // child is not done inferencing
// default handler for signal SIGINT
void handle_SIGUSR2(int signum){
    // Update the SIGUSR2 flag
    SIGUSR2_flag = 1;
}
void handle_SIGINT(int signnum){
    printf("SIGINT i.e. CTRL-C Received");
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
    // Write your main logic here
    int pid;
    int pfd[2]; // pipe file descriptor
    if (pipe(pfd) == -1){
        printf("Pipe Error");
        return 2;
    } else {printf("Pipe Created Successfully\n");}

    // install signal handler for SIGUSR2 and SIGINT
    struct sigaction SIGUSR2_handler, SIGINT_handler;
    SIGUSR2_handler.sa_handler = handle_SIGUSR2;
    SIGINT_handler.sa_handler = handle_SIGINT;
    sigemptyset(&SIGUSR2_handler.sa_mask);
    sigemptyset(&SIGINT_handler.sa_mask);
    SIGUSR2_handler.sa_flags = SA_RESTART;
    SIGINT_handler.sa_flags = SA_RESTART;

    // Measure the resources
    struct rusage used;

    // use fork to create child process 
    pid = fork();
    if (pid == 0){
        printf("Child process");
        close(pfd[WRITE_END]); // close the write end for child process
        if (dup2(pfd[READ_END], READ_END) == -1){ // set pipe read end to stdin
            perror("dup2 failed");} else {printf("Child process dup2");}
        close(pfd[READ_END]); // close the pipe read end
        if (execlp("./inference", "./inference", seed, NULL) == -1){ // use exec to run inference_[UID].c for child process 
            perror("execlp failed");
            exit(1);}
    } 
    // main process: 
    // get user prompt -> pass to inference process
    // -> wait until the inference process finish inferencing & access to /proc to retrieve cpu usage information
    else { 
        // in the main process, accept user input
        sleep(3);
        char buf[MAX_PROMPT_LEN];
        int status;
        close(pfd[READ_END]); // close read end for main process
        // accept the user prompt up to 4 or until the SIGINT is received
        for(int i = 0; i < 4; i++){ // run until SIGINT not received or num_prompt < 4
            printf(">>> ");
            if (fgets(buf, MAX_PROMPT_LEN, stdin) == NULL){printf("EOF Error");} 
            else{
                printf("User prompt received \n");
                printf("%s", buf);
                }
            int pipe_write = write(pfd[WRITE_END], buf, strlen(buf));
            printf("%d", pipe_write);
            // pass the prompt to the pipeline to the child process.
            if (pipe_write == -1){
                perror("Main process failed to write on child child process");
            } else {printf("User prompt written on the pipe successfully. \n");}
            kill(pid, SIGUSR1); // send SIGUSR1 to child process to notice the user prompt is ready
            printf("Signal SIGUSR1 Sent to Child Process and Entering to the while loop \n");
            // while the child process is inferencing
            // Monitoring status of inference process
            while(!SIGUSR2_flag){
                sprintf(path, "/proc/%d/stat", pid);
                FILE *fp = fopen(path, "r");
                if (fp == NULL){
                    fprintf(stderr, "Opening /proc/{pid}/stat Failed: %s\n", strerror(errno));
                }
                sleep(0.3); // sleep for 300ms 
            }
            // reset the SIGUSR2 flag
            SIGUSR2_flag = 0;
        }   
        // close the write end
        close(pfd[WRITE_END]);
        wait4(pid, &status, 0, &used); // wait for child process to finish
        printf("Child process exited, with exit status: %d\n", WTERMSIG(status));
    }

    return EXIT_SUCCESS;
}