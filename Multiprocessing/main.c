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
#include <sys/resource.h>
#include <time.h>

// install signal handler here
int SIGUSR2_flag = SYSCALL_FLAG; // child is not done inferencing
// default handler for signal SIGINT
void handle_SIGUSR2(int signum){
    // Update the SIGUSR2 flag
    SIGUSR2_flag = 1;
}

void handle_SIGINT(int signum){
    fprintf(stderr, "SIGINT i.e. CTRL-C Received\n");
    exit(130);
}

// file path for /proc/pid/meminfo
char path[256]; 

struct stat {
    int pid;
    char tcomm[256];
    char state;
    unsigned long current_utime;
    unsigned long current_stime;
    unsigned long last_utime;
    unsigned long last_stime;
    double last_tstamp_s;
    long nice;
    unsigned long vsize;
    int processor;
    unsigned int policy;    
};
// Monitor item: 
/*
pid: 1 %d
tcomm: 2 %s
state: 3 %c
policy 41 %u
nice: 19 %ld
vsize: 23 %lu
task_cpu - processor: 39 %d  
utime: 14 %lu
stime: 15 %lu
*/

static inline double now_monotonic_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int Monitor(char path[], struct stat *out){
    FILE *fp = fopen(path, "r");
    if (!fp){
        fprintf(stderr, "Opening /proc/{pid}/stat Failed\n");
    }
    if (fscanf(fp, "%d (%255[^)]) %c",
               &out->pid, out->tcomm, &out->state) != 3) {
        fprintf(stderr, "parse header failed for %s\n", path);
        fclose(fp);
        return -1;
    }
    // 2) Scan only the desired tail fields, skipping others with %*
    if (fscanf(fp,
        // 4..8  ppid pgrp session tty_nr tpgid
        " %*d %*d %*d %*d %*d"
        // 9..13 flags minflt cminflt majflt cmajflt
        " %*u %*u %*u %*u %*u"
        // 14..15 utime stime
        " %lu %lu"
        // 16..19 cutime cstime priority nice
        " %*d %*d %*d %ld"
        // 20..22 num_threads itrealvalue starttime
        " %*d %*d %*u"
        // 23..24 vsize rss
        " %lu %*d"
        // 25..31 rsslim startcode endcode startstack kstkesp kstkeip signal
        " %*u %*u %*u %*u %*u %*u %*u"
        // 32..34 blocked sigignore sigcatch
        " %*u %*u %*u"
        // 35..37 wchan nswap cnswap
        " %*u %*u %*u"
        // 38 exit_signal
        " %*d"
        // 39..41 processor rt_priority policy
        " %d %*u %u",
        &out->current_utime, &out->current_stime, &out->nice,
        &out->vsize, &out->processor, &out->policy) != 6) {
        fprintf(stderr, "parse tail failed for %s\n", path);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}
// Function using SYS_sched_setattr syscll to set the scheduling policy and related scheduling parameters for the inference child process
void 
struct stat s;

int main(int argc, char *argv[]) {
    char* seed; 
    if (argc == 2) {
        seed = argv[1];
    } else if (argc == 1) {
        // use 42, the answer to life the universe and everything, as default
        seed = "42";
    } //else if (argc == 3) {
//        if (strcmp(argv[2], "2>log") == 0){
//            monitor_flag = 1;
//        }
//    } 
    else {
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
    } // else {fprintf(stderr, "Pipe Created Successfully\n");}

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

    // use fork to create child process 
    pid = fork();
    if (pid == 0){
        close(pfd[WRITE_END]); // close the write end for child process

        if (dup2(pfd[READ_END], STDIN_FILENO) == -1){ // set pipe read end to stdin
            perror("dup2 failed");
            return 2;} // else {fprintf(stderr, "Child process dup2\n");}
        close(pfd[READ_END]); // close the read end for child process
        if (execl("./inference", "inference", seed, (char *)NULL) == -1){ // use execl to run inference_[UID].c for child process 
            perror("execlp failed");
            return 3;}
    } 
    // main process: 
    // get user prompt . pass to inference process
    // . wait until the inference process finish inferencing & access to /proc to retrieve cpu usage information
    else { 
        // in the main process, accept user input
        // sleep(3);
        char buf[MAX_PROMPT_LEN];
        int status;
        pid_t w;
        close(pfd[READ_END]); // close read end for main process
        // accept the user prompt up to 4 or until the SIGINT is received
        for(int i = 0; i < 4; i++){ // run until SIGINT not received or num_prompt < 4
            printf(">>> ");
            fflush(stdout);
            if(fgets(buf, MAX_PROMPT_LEN, stdin) == NULL){
                fprintf(stderr, "EOF Error\n");
                return 4;
            } 
            // Potential Error Occured here
            if(write(pfd[WRITE_END], buf, strlen(buf)) == -1){
                fprintf(stderr, "Pipe Write Error\n");
                return 5;
            }
            kill(pid, SIGUSR1); // send SIGUSR1 to child process to notice the user prompt is ready
            // while the child process is inferencing
            
            // Monitoring status of inference process
            
            sprintf(path, "/proc/%d/stat", pid);
            while(!SIGUSR2_flag){
                usleep(300000);// sleep for 300 ms
                // Monitor the /proc file system only when 2>log
                if (Monitor(path, &s) == 0){
                    double t_now = now_monotonic_s();
                    double dt = t_now - s.last_tstamp_s;
                    if (dt <= 0) dt = 1e-9; // guard tiny/zero interval

                    double cpu_pct = ((s.current_utime - s.last_utime) + (s.current_stime - s.last_stime)) / dt * 100;
                    s.last_utime = s.current_utime; s.last_stime = s.current_stime;
                    fprintf(stderr, "[pid] %d [tcomm] %s [state] %c [policy] %u [nice] %ld [vsize] %lu [task_cpu] %d [utime] %lu [stime] %lu [cpu%%] %.3f%%\n", 
                    s.pid, s.tcomm, s.state, s.policy, s.nice, s.vsize, s.processor, s.current_utime, s.current_stime, cpu_pct);
                };
            }
            // reset the SIGUSR2 flag
            SIGUSR2_flag = 0;
        }
        w = waitpid(pid, &status, 0); // wait for child process to finish
        if(w == -1){
            fprintf(stderr, "Waitpid Error\n");
        }
        fprintf(stderr, "Child process exited, with exit status: %d\n", WEXITSTATUS(status));
        close(pfd[WRITE_END]);
    }
    return EXIT_SUCCESS;
}