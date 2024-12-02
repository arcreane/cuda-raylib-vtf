#include "utils.hpp"
#include <raylib.h> // For GetFrameTime

// Generates a random float between min and max
float random_float(float min, float max) {
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

// Returns the time elapsed for the current frame
float calculate_frame_time() {
    return GetFrameTime(); // Uses Raylib's built-in function
}
