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
// file path for /proc/pid/meminfo
char path[256]; 

// CPU usage and Memory usage of child process
char cpuinfo[256];
char meminfo[256];
// install signal handler here
int SIGUSR2_flag = SYSCALL_FLAG; // child is not done inferencing
int count = 0;
// default handler for signal SIGINT
void handle_SIGUSR2(int signum){
    // Update the SIGUSR2 flag
    SIGUSR2_flag = 1;
    fprintf(stderr, "Main process: SIGUSR2 Received\n");
    count++;
}
void handle_SIGINT(int signnum){
    fprintf(stderr, "SIGINT i.e. CTRL-C Received\n");
    exit(1);
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
        fprintf(stderr, "Pipe Error");
        return 2;
    } else {fprintf(stderr, "Pipe Created Successfully\n");}

    // install signal handler for SIGUSR2 and SIGINT
    struct sigaction SIGUSR2_handler, SIGINT_handler;

    SIGUSR2_handler.sa_handler = handle_SIGUSR2;
    SIGINT_handler.sa_handler = handle_SIGINT;
 
    sigemptyset(&SIGUSR2_handler.sa_mask);
    sigemptyset(&SIGINT_handler.sa_mask);

    SIGUSR2_handler.sa_flags = SA_RESTART;
    SIGINT_handler.sa_flags = SA_RESTART;

    sigaction(SIGUSR2, &SIGUSR2_handler, NULL);
    sigaction(SIGINT, &SIGINT_handler, NULL);
    // Measure the resources
    struct rusage used;

    // use fork to create child process 
    pid = fork();
    if (pid == 0){
        fprintf(stderr, "Child process begin\n");
        close(pfd[WRITE_END]); // close the write end for child process

        if (dup2(pfd[READ_END], STDIN_FILENO) == -1){ // set pipe read end to stdin
            perror("dup2 failed");
            return 2;} else {fprintf(stderr, "Child process dup2\n");}
        close(pfd[READ_END]); // close the read end for child process
        if (execl("./inference", "inference", seed, (char *)NULL) == -1){ // use execl to run inference_[UID].c for child process 
            perror("execlp failed");
            return 3;}
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
            fflush(stdout);
            if(fgets(buf, MAX_PROMPT_LEN, stdin) == NULL){
                perror("EOF Error");
                return 4;
            } else{
                fprintf(stderr, "User prompt received \n");
                fprintf(stderr, "%s", buf);
            }
            // Potential Error Occured here
            if(write(pfd[WRITE_END], buf, strlen(buf)) == -1){
                perror("Pipe Write Error");
                return 5;
            }else{
                fprintf(stderr, "Main process Write to Child Process\n");
            }
            kill(pid, SIGUSR1); // send SIGUSR1 to child process to notice the user prompt is ready
            fprintf(stderr, "Signal SIGUSR1 Sent to Child Process and Entering to the while loop \n");
            // while the child process is inferencing
            // Monitoring status of inference process
            FILE * cpu_fp = fopen("/proc/cpuinfo", "r");
            if(cpu_fp == NULL){
                fprintf(stderr, "CPU Retrieving Failed");
            }
            sprintf(path, "/proc/%d/meminfo", pid);
            FILE *mem_fp = fopen(path, "r");
            if (mem_fp == NULL){
                fprintf(stderr, "Opening /proc/{pid}/meminfo Failed\n");
            }
            while(!SIGUSR2_flag){
                usleep(300000);// sleep for 300 ms
                // Monitoring CPU usage and Memory usage of inference process
                if(fgets(cpuinfo, sizeof(cpuinfo), cpu_fp) == NULL){
                    fprintf(stderr, "Reading CPU usage Error\n");
                } else{
                    fprintf(stderr, "%s\n", cpuinfo);
                }
            //    if(fgets(meminfo, sizeof(meminfo), mem_fp) == NULL){
     //               fprintf(stderr, "Reading Memory usage Error\n");
 //               } else {
               //     fprintf(stderr, "%s\n", meminfo);
     //           }
            }
            fclose(cpu_fp);
            fclose(mem_fp);
            // reset the SIGUSR2 flag
            SIGUSR2_flag = 0;
        }
        // close the write end
        fprintf(stderr, "Main Process: 4 Prompt Received. Now Close the write end\n");
        close(pfd[WRITE_END]);
        fprintf(stderr, "Main Process: Wait for Child Process to terminate");
        wait4(pid, &status, 0, &used); // wait for child process to finish
        fprintf(stderr, "Child process exited, with exit status: %d\n", WTERMSIG(status));
    }

    return EXIT_SUCCESS;
}