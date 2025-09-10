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

int SIGUSR2_flag = SYSCALL_FLAG; // child is not done inferencing

void handle_SIGUSR2(int signum){
    // Update the SIGUSR2 flag
    SIGUSR2_flag = 1;
}

void handle_SIGINT(int signum){
    fprintf(stderr, "SIGINT i.e. CTRL-C Received\n");
    exit(130);
}

char path[256]; 
struct stat {
    int pid;
    char tcomm[256];
    char state;
    unsigned long current_utime;
    unsigned long current_stime;
    unsigned long last_utime;
    unsigned long last_stime;
    unsigned long last_tstamp_s;
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
struct stat s;
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
struct sched_attr attr; // from common.h
int policy, niceval, rt_prio;
// Read current policy, nice, and RT priority via raw syscall
int get_policy_nice_raw(pid_t pid, int *policy, int *niceval, int *rt_prio) {
    struct sched_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;

    long rc = syscall(SYS_sched_getattr, pid, &a, sizeof a, 0u);
    if (rc < 0) return -1;

    if (policy)  *policy  = (int)a.sched_policy;
    if (niceval) *niceval = (int)a.sched_nice;        // meaningful for OTHER/BATCH/IDLE
    if (rt_prio) *rt_prio = (int)a.sched_priority;    // meaningful for FIFO/RR
    printf("[Scheduling Policy, Nice value, Priority]\n");
    printf("[pid] %d, [policy] %s [nice] %d [priority] %u\n", pid, get_sched_name((int)a.sched_policy), (int)a.sched_nice, (unsigned)a.sched_priority);
    return 0;
}

// Set policy + nice (or RT priority) for a target pid via raw syscall
// policy: SCHED_OTHER/BATCH/IDLE/FIFO/RR
// rt_prio: 1..99 for FIFO/RR (ignored for OTHER/BATCH/IDLE)
// niceval: -20..19 for time-sharing policies (ignored by RT policies)
int set_policy_nice_raw(pid_t pid, int policy, int niceval, int rt_prio) {
    struct sched_attr a;
    memset(&a, 0, sizeof a);
    a.size         = sizeof a;
    a.sched_policy = policy;

    if (policy == SCHED_FIFO || policy == SCHED_RR) {
        if (rt_prio < 1 || rt_prio > 99) { printf("Invalid Priority Value\n"); return -1; }
        a.sched_priority = (unsigned)rt_prio;   // needs CAP_SYS_NICE
        a.sched_nice     = niceval;             // harmless but ignored by RT
    } else {
        if (niceval < -20 || niceval > 19) { printf("Invalid Nice Value\n"); return -1; }
        a.sched_nice = niceval;                 // used by OTHER/BATCH/IDLE
    }
    // Third argument (flags) usually 0. Leave a.sched_flags=0 unless you specifically need
    // SCHED_FLAG_RESET_ON_FORK in attr.sched_flags for children.
    long rc = syscall(SYS_sched_setattr, pid, &a, 0u);
    if (rc < 0) return -1;
    printf("Policy, Nice value and Priority has been updated.\n");
    return 0;
}

int main(int argc, char *argv[]) {
    char* seed; 
    if (argc == 2) {
        seed = argv[1];
    } else if (argc == 1) {
        // use 42, the answer to life the universe and everything, as default
        seed = "42";
    } 
    else {
        fprintf(stderr, "Usage: ./main <seed>\n");
        fprintf(stderr, "Note:  default seed is 42\n");
        exit(1);
    }

    int pid;
    int pfd[2]; // pipe file descriptor
    if (pipe(pfd) == -1){
        fprintf(stderr, "Pipe Error");
        return 2;
    } 

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

    pid = fork();
    if (pid == 0){
        close(pfd[WRITE_END]); // close the write end for child process
        if (dup2(pfd[READ_END], STDIN_FILENO) == -1){ // set pipe read end to stdin
            perror("dup2 failed");
            return 2;} 
        close(pfd[READ_END]); // close the read end for child process
        if (execl("./inference", "inference", seed, (char *)NULL) == -1){ // use execl to run inference_[UID].c for child process 
            perror("execlp failed");
            return 3;}
    } 
    else { 
        char buf[MAX_PROMPT_LEN];
        int status;
        pid_t w;
        close(pfd[READ_END]); // close read end for main process
        // set scheduling policy and niceval 
        get_policy_nice_raw(pid, &policy, &niceval, &rt_prio);
        printf("Please enter policy, nice, priority respectively: ");
        scanf("%d %d %d", &policy, &niceval, &rt_prio);
        /* consume everything up to and including the newline */
        int ch;
        while ((ch = getchar()) != '\n' && ch != EOF) { /* nothing */ }
        set_policy_nice_raw(pid, policy, niceval, rt_prio);
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
            // Monitoring status of inference process        
            sprintf(path, "/proc/%d/stat", pid);
            while(!SIGUSR2_flag){
                usleep(300000);// sleep for 300 ms
                if (Monitor(path, &s) == 0){
                    double t_now = time_in_ms();
                    double dt = t_now - s.last_tstamp_s;
                    s.last_tstamp_s = t_now;
                    if (dt <= 0) dt = 1e-9; // guard tiny/zero interval
                    double cpu_pct = ((s.current_utime - s.last_utime) + (s.current_stime - s.last_stime)) / dt * 100;
                    s.last_utime = s.current_utime; 
                    s.last_stime = s.current_stime;
                    fprintf(stderr, "[pid] %d [tcomm] %s [state] %c [policy] %s [nice] %ld [vsize] %lu [task_cpu] %d [utime] %lu [stime] %lu [cpu%%] %.3f%%\n", 
                    s.pid, s.tcomm, s.state, get_sched_name(s.policy), s.nice, s.vsize, s.processor, s.current_utime, s.current_stime, cpu_pct);
                };
            }
            // reset the SIGUSR2 flag
            SIGUSR2_flag = 0;
        }
        w = waitpid(pid, &status, 0); // wait for child process to finish
        if(w == -1){
            fprintf(stderr, "Waitpid Error\n");
        }
        printf("Child process exited, with exit status: %d\n", WEXITSTATUS(status));
        close(pfd[WRITE_END]);
    }
    return EXIT_SUCCESS;
}