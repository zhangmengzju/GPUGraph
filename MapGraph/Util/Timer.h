#ifndef TIMER_H_DEF
#define TIMER_H_DEF

#ifdef WIN32   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif


class Timer {
public:
    Timer();
    ~Timer();
    void   start();
    void   stop();
    double getElapsedTimeInSec();
    double getElapsedTimeInMilliSec();
    double getElapsedTimeInMicroSec();

private:
    double startTimeInMicroSec;
    double endTimeInMicroSec;
    int    stopped;
    #ifdef WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER startCount;
    LARGE_INTEGER endCount;
    #else
    timeval startCount;timeval endCount;                           //

#endif
};

#endif // TIMER_H_DEF
