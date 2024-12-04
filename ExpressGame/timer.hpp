#ifndef TIMER_HPP
#define TIMER_HPP

#include "raylib.h"
#include <string>

class Timer {
private:
    float timeLeft; // Time left for the countdown

public:
    // Constructor to initialize the timer with a specific duration
    Timer(float initialTime);

    // Updates the timer, decrementing based on frame time
    void Update();

    // Returns the time left as a formatted string
    std::string GetTimeLeft() const;

    // Checks if the timer has reached zero
    bool IsTimeUp() const;
};

#endif // TIMER_HPP
