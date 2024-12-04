#include "Timer.hpp"
#include <cstdio>

// Constructor: Initializes the timer with a given duration
Timer::Timer(float initialTime) : timeLeft(initialTime) {}

// Updates the timer based on the time elapsed since the last frame
void Timer::Update() {
    timeLeft -= GetFrameTime();
    if (timeLeft < 0) timeLeft = 0; // Clamp to 0
}

// Returns the remaining time as a formatted string
std::string Timer::GetTimeLeft() const {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "Time Left: %.1f seconds", timeLeft);
    return std::string(buffer);
}

// Checks if the timer has reached zero
bool Timer::IsTimeUp() const {
    return timeLeft <= 0;
}
